import os
import argparse
import torch
import numpy
import math
import logging

import data_util
from model_util import cap2img_grounding
from tiger import Evaluator, encode_cands

from SCAN.data import get_test_loader
from SCAN.vocab import Vocabulary, deserialize_vocab
from SCAN.evaluation import encode_data


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate_path', default='./data/candidates/',
                        help='path to testing datasets')
    parser.add_argument('--data_path', default='./data/precomp/',
                        help='path to testing datasets')
    parser.add_argument('--data_name', default='precomp',
                        help='{composite_coco, composite_8k, composite_30k, flickr8k, pascal}_precomp')
    parser.add_argument('--split_name', default='cand',
                        help='prefix of reference files')
    parser.add_argument('--output_path', default='./data/output/',
                        help='path to output path')
    parser.add_argument('--scan_opt_path', default='./scan_opt.pkl',
                        help='path to scan opt file')
    parser.add_argument('--vocab_path', default='./vocab/coco_precomp_vocab.json',
                        help='path to vocab file')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size')
    parser.add_argument('--refsample_size', default=None, type=int,
                        help='number of reference captions to be considered')
    opt = parser.parse_args()

    # Load Vocabulary Wrapper
    vocab = deserialize_vocab(opt.vocab_path)

    # Setup Evaluator
    evaluator = Evaluator(opt.data_path, opt.data_name, opt.split_name, vocab, opt.scan_opt_path)

    # Load test data
    gts_img, gts_ref, cands = load_test_data(opt, evaluator, vocab)

    # Begin evaluation
    evaluator.tiger_score(gts_img, gts_ref, cands, opt.refsample_size, opt.output_path)



def load_test_data(opt, evaluator, vocab):
    # Load ground truth data
    scan_opt = evaluator.scan_opt
    model = evaluator.scan_model
    test_loader = get_test_loader(opt.split_name, opt.data_name, vocab, opt.batch_size, scan_opt.workers, scan_opt)
    gts_img, gts_ref = get_gts(model, scan_opt, opt.split_name, test_loader)

    # Get candidate captions
    if opt.data_name.startswith('composite'):
        return gts_img, gts_ref, get_cands_composite(model, vocab, opt)
    elif opt.data_name.startswith('flickr8k'):
        return gts_img, gts_ref, get_cands_flickr8k(model, vocab, opt)
    elif opt.data_name.startswith('pascal'):
        return gts_img, gts_ref, get_cands_pascal(model, vocab, opt)
    elif opt.data_name.startswith('usecase'):
        return gts_img, gts_ref, get_cands_usecase(model, vocab, opt)
    else:
        print('opt.data_name = {} is not supported. Avaliable test sets include [composite, flickr8k, pascal, usecase].'.format(opt.data_name))
        exit(0)


def get_gts(model, scan_opt, split_name, val_loader):
    img_embs, ref_embs, ref_lens = encode_data(model, val_loader, scan_opt.log_step, logging.info)
    ref_strings = data_util.get_ref_capstrings(scan_opt)
    imgids = data_util.cap2imgid(scan_opt, split_name)
    gts_img = {}
    gts_ref = {}
    for k in range(len(imgids)):
        imgid = imgids[k]
        if imgid not in gts_img.keys():
            gts_img[imgid] = img_embs[k]
        if imgid not in gts_ref.keys():
            gts_ref[imgid] = []
        item = {}
        item['emb'] = ref_embs[k]
        item['len'] = ref_lens[k]
        item['string'] = ref_strings[k]
        gts_ref[imgid].append(item)
    del img_embs, ref_embs, ref_lens, ref_strings, imgids
    torch.cuda.empty_cache()
    return gts_img, gts_ref


def get_cands_usecase(model, vocab, opt):
    cands = {}
    imgids, caps_string, caps_tensor, caps_lengths = data_util.read_usecase(vocab, opt)
    caps_embs, caps_lens = encode_cands(model, caps_tensor, caps_lengths)
    for k in range(len(imgids)):
        imgid = imgids[k]
        if imgid not in cands.keys():
            cands[imgid] = []
        item = {}
        item['emb'] = caps_embs[k]
        item['len'] = caps_lens[k]
        item['string'] = caps_string[k]
        cands[imgid].append(item)
    return cands


def get_cands_composite(model, vocab, opt):
    cands = {}
    cands_split = opt.data_name.split('_')[-2]
    caps = data_util.read_composite(vocab, opt, cands_split)
    for sys, val in caps.items():
        caps_embs, caps_lens = encode_cands(model, val['tensor'], val['length'])
        caps[sys]['emb'] = caps_embs
        caps[sys]['length'] = caps_lens
        for k in range(len(caps[sys]['imgid'])):
            imgid = caps[sys]['imgid'][k]
            if imgid not in cands.keys():
                cands[imgid] = []
            item = {}
            item['emb'] = caps[sys]['emb'][k]
            item['len'] = caps[sys]['length'][k]
            item['string'] = caps[sys]['string'][k]
            item['type'] = sys
            item['eval'] = caps[sys]['eval'][k]
            cands[imgid].append(item)
    del caps
    return cands

def get_cands_flickr8k(model, vocab, opt):
    cands = {}
    imgids, caps_string, caps_tensor, caps_lengths, caps_evals, caps_id, caps_idxs = data_util.read_flickr8k(vocab, opt)
    caps_embs, caps_lens = encode_cands(model, caps_tensor, caps_lengths)
    for k in range(len(imgids)):
        imgid = imgids[k]
        if imgid not in cands.keys():
            cands[imgid] = []
        item = {}
        item['emb'] = caps_embs[k]
        item['len'] = caps_lens[k]
        item['string'] = caps_string[k]
        item['eval'] = caps_evals[k]
        cands[imgid].append(item)
    return cands


def get_cands_pascal(model, vocab, opt):
    cands = {}
    imgids, caps_string, caps_tensor, caps_lengths, caps_type, caps_idxs, caps_pairevals, caps_pairtype, caps_pairid = data_util.read_pascal(vocab, opt)
    num_iter = int(math.ceil(1.0 * len(imgids)/opt.batch_size))

    for i in range(num_iter):
        st,end = (i * opt.batch_size, min(i* opt.batch_size + opt.batch_size, len(imgids)))
        caps_embs, caps_lens = encode_cands(model, caps_tensor[st:end], caps_lengths[st:end])
        for k in range(st,end):
            imgid = imgids[k]
            if imgid not in cands.keys():
                cands[imgid] = []
            item = {}
            item['emb'] = caps_embs[k%opt.batch_size]
            item['len'] = caps_lens[k%opt.batch_size]
            item['string'] = caps_string[k]
            item['eval'] = caps_pairevals[k]
            item['pairid'] = caps_pairid[k]
            item['pairtype'] = caps_pairtype[k]
            item['captype'] = caps_type[k]
            cands[imgid].append(item)

    del imgids, caps_string, caps_tensor, caps_lengths, caps_type, caps_idxs, caps_pairevals, caps_pairtype, caps_pairid, caps_embs, caps_lens
    torch.cuda.empty_cache()
    return cands

if __name__ == '__main__':
    main()
