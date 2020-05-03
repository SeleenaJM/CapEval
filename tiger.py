from __future__ import print_function
import os
import math
import random
import pickle 
import torch
import numpy as np
from torch.autograd import Variable
from scipy import spatial
from argparse import Namespace

from model_util import cands_forward_emb, cap2img_grounding
from data_util import cap2imgid, flickr8k_imgname2idx
from SCAN.model import SCAN

class Evaluator(object):
    def __init__(self, data_path, data_name, split_name, vocab, scan_opt_path='./scan_opt.pkl'):
        # load pre-trained scan model      
        sopt = pickle.load(open(scan_opt_path, 'rb'))
        sopt.data_path = data_path
        sopt.data_name = data_name
        sopt.split_name = split_name
        sopt.vocab_size = len(vocab)
        self.scan_model = SCAN(sopt)
        if os.path.isfile(sopt.resume):
            print("=> loading checkpoint '{}'".format(sopt.resume))
            checkpoint = torch.load(sopt.resume)
            self.scan_model.load_state_dict(checkpoint['model'])
        else:
            print("A pre-trained SCAN model is required.")
            exit(0)
        self.scan_opt = sopt
        self.vocab = vocab

    def tiger_score(self, gts_img, gts_ref, cands, ref_sample=None, output_dir=None):
        """
        Args:
            img_feat_file: path to the extracted image features in numpy format
            ref_strings: (N, NUM_REF, MAX_CAP_LEN), list of reference caption strings
            cand_strings: (N, NUM_CAND, MAX_CAP_LEN), list of candidate caption strings
        Return:
            scores: (imgid, cap string, tiger score, human score, system id), list of scores if evaluation is a rating scale
                    or
                    {pairid {imgid, pair type, system_B cap string, system_C cap string, 
                             system_B tiger score, system_C tiger score, human score}}, dictionary of scores if evaluation is a pairwise comparision 
        """
        # compute scores in a loop of N images

        if not self.scan_opt.data_name.startswith('pascal'): 
            scores = []
            if self.scan_opt.data_name.startswith('composite_8k') or self.scan_opt.data_name.startswith('flickr8k'):
                ref_imgids, ref_imgnames = flickr8k_imgname2idx(self.scan_opt.data_path, self.scan_opt.data_name, self.scan_opt.split_name)
        else:
            scores = {}

        for imgid in cands.keys():
            if self.scan_opt.data_name.startswith('composite_8k') or self.scan_opt.data_name.startswith('flickr8k'):
                sudo_imgid = ref_imgids[ref_imgnames.index(imgid)]
                imgfeat = Variable(torch.from_numpy(gts_img[sudo_imgid]), volatile=True).cuda()
                refs = gts_ref[sudo_imgid]
            else:
                imgfeat = Variable(torch.from_numpy(gts_img[imgid]), volatile=True).cuda()
                refs = gts_ref[imgid]
            caps = cands[imgid]
            refs_sim = []
            for r in refs:
                rcap = Variable(torch.from_numpy(r['emb']), volatile=True).cuda()
                rlen = r['len']
                rsim = cap2img_grounding(imgfeat, rcap, rlen, self.scan_opt)
                refs_sim.append(rsim)
            #Randomly sample k references (k = ref_sample)
            if ref_sample:
                refs_sim = random.sample(refs_sim, ref_sample)

            for c in caps:
                if 'type' in c.keys() and c['type'] == 'h':
                    stacked_tensor = torch.stack(refs_sim[1:])
                else:
                    stacked_tensor = torch.stack(refs_sim)

                mean_ref2img_sim =  torch.t(stacked_tensor).mean(dim=1, keepdim=True)    
                mean_ref2img_sim = mean_ref2img_sim.tolist()
                mean_ref2img_sim = [i[0] for i in mean_ref2img_sim]

                ccap = Variable(torch.from_numpy(c['emb']), volatile=True).cuda()
                clen = c['len']
                c2img_sim = cap2img_grounding(imgfeat, ccap, clen, self.scan_opt).tolist()

                dds = ndcg(mean_ref2img_sim, c2img_sim)
                kld = kld_softmax(mean_ref2img_sim, c2img_sim)
                d_rel = absolute_distance(mean_ref2img_sim, c2img_sim)
                wds = kld + d_rel
                sigmoid_wds = 1 - sigmoid(wds, 1.5)
                tiger_score = 0.5 * dds + 0.5 * sigmoid_wds

                #if not self.scan_opt.data_name.startswith('pascal'):
                if self.scan_opt.data_name.startswith('usecase'):
                    scores.append((imgid, c['string'], tiger_score))
                elif self.scan_opt.data_name.startswith('composite') or self.scan_opt.data_name.startswith('flickr'):
                    scores.append((imgid, c['string'], tiger_score, c['eval']))
                else:
                    if c['pairid'] not in scores.keys():
                        scores[c['pairid']] = {}
                    scores[c['pairid']]['imgid'] = imgid
                    scores[c['pairid']]['pairtype'] = c['pairtype']
                    scores[c['pairid']]['eval'] = 'B' if c['eval'] == '1' else 'C'
                    if c['captype'] == 'B':
                        scores[c['pairid']]['B_string'] = c['string'] 
                        scores[c['pairid']]['B_score'] = tiger_score 
                    else:
                        scores[c['pairid']]['C_string'] = c['string']
                        scores[c['pairid']]['C_score'] = tiger_score                    
        if output_dir:
            file_name = self.scan_opt.data_name.replace('_precomp', '_score')
            self.print_score(scores, output_dir, file_name, ref_sample)


    def print_score(self, scores, output_dir, file_name, ref_sample=None):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)  
        if ref_sample:
            output_file = os.path.join(output_dir, file_name+'_refsize_'+str(ref_sample)+'.csv') 
        else:
            output_file = os.path.join(output_dir, file_name+'.csv')
        if not os.path.exists(output_file):
            f=open(output_file, 'a+')
            if file_name.startswith('usecase'):
                f.write("imgid\tcaption\tTIGER\n")
            elif file_name.startswith('composite'):
                f.write("imgid\tcaption\tTIGER\tHUMAN\n")
            elif file_name.startswith('flickr8k'):
                f.write("imgid\tcaption\tTIGER\tHUMAN 1\tHUMAN 2\tHUMAN 3\n")
            else:
                f.write("imgid\tpair type\tcaption B\tcaption C\tTIGER B\tTIGER C\tTIGER Pariwise\tHUMAN Pairwise\n")
        else:
            #f = open(output_file, 'a+')
            print(file_name, 'exists! Please delete the file!')
        
        towrite = ''
        
        if file_name.startswith('usecase'):
            overall_tigerscore = 0.0
            for item in scores:
                towrite += str(item[0]) +'\t'+ str(item[1]) +'\t'+ str(item[2]) + '\n'
                overall_tigerscore += item[2]
            overall_tigerscore = 1.0*overall_tigerscore/len(scores)
            print('\n\nThe overall system score is: ', str(overall_tigerscore))
        elif file_name.startswith('composite'):
            for item in scores:
                towrite += str(item[0]) +'\t'+ str(item[1]) +'\t'+ str(item[2]) +'\t'+ str(item[3]) + '\n'
        elif file_name.startswith('flickr8k'):
            for item in scores:
                towrite += str(item[0]) +'\t'+ str(item[1]) +'\t'+ str(item[2]) +'\t'+ str('\t'.join(item[3])) + '\n'
        elif file_name.startswith('pascal'):
            for k,v in scores.items():
                if v['B_score'] > v['C_score']:
                    pairwise_result = 'B' 
                elif v['B_score'] < v['C_score']:
                    pairwise_result = 'C'
                else:
                    pairwise_result = 'Equal'
                towrite += str(v['imgid']) +'\t'+ str(v['pairtype']) +'\t'+ str(v['B_string']) +'\t'+ str(v['C_string']) +'\t'+ str(v['B_score']) +'\t'+ str(v['C_score']) +'\t'+ str(pairwise_result) +'\t'+ str(v['eval']) +'\n'
        f.write(towrite)
        f.close()



def encode_cands(model, cap_tensors, lengths):    
    model.val_start()
    np_cap_embs = None
    cap_lens = None
    max_n_word = 0
    max_n_word = max(max_n_word, max(lengths))
    cap_embs, cap_lens = cands_forward_emb(model, cap_tensors, lengths, volatile=True)   
    if np_cap_embs is None:
        np_cap_embs = np.zeros((cap_embs.size(0), max_n_word, cap_embs.size(2)))
    for i, cap_emb in enumerate(cap_embs):
        np_cap_embs[i] = cap_emb.data.cpu().numpy().copy()
    
    del cap_tensors, cap_embs
    return np_cap_embs, cap_lens


def ndcg(ref, test):
    norm_ref = [((x - min(ref)) / (max(ref) - min(ref))) for x in ref]
    norm_test = [((y - min(test)) / (max(test) - min(test))) for y in test]
    sorted_ref_idx = [b[0] for b in sorted(enumerate(norm_ref),key=lambda i:i[1], reverse = True)]
    sorted_test_idx = [c[0] for c in sorted(enumerate(norm_test),key=lambda i:i[1], reverse = True)]

    idcg = 0.0
    for i in range(0, len(sorted_ref_idx)):
        idx = sorted_ref_idx[i]
        if i == 0:
            idcg = norm_ref[idx]
        else:
            idcg += norm_ref[idx]/(math.log(i + 1 + 1, 2))
    dcg = 0.0
    for i in range(0, len(sorted_test_idx)):
        idx = sorted_test_idx[i]
        if i == 0:
            dcg = norm_ref[idx]
        else:
            dcg += norm_ref[idx]/(math.log(i + 1 + 1, 2))
    ndcg = dcg/idcg
    return ndcg


def sigmoid(val, tau):
    sig_val = math.exp(tau * val)/(math.exp(tau * val) + 1)
    return sig_val

def absolute_distance(ref, test):
    ref_mode = math.sqrt(sum([i*i for i in ref]))
    cand_mode = math.sqrt(sum([i*i for i  in test]))
    if cand_mode == 0.0 or ref_mode == 0.0:
        dis = 0.0
    else:
        dis = np.log(ref_mode/cand_mode)
    return dis


def kld_softmax(ref, test):
    kl_diver = 0.0
    ref_soft = self_softmax(ref, 1.0)
    test_soft = self_softmax(test, 1.0)
    for idx in range(0, len(ref_soft)):
        p_t_ref = ref_soft[idx]
        p_t_test = test_soft[idx]
        if p_t_ref == 0.0 or p_t_test == 0.0:
            kl_diver += 0.0
        else:
            kl_diver += p_t_ref * np.log(p_t_ref/p_t_test)
    return kl_diver


def self_softmax(vector, beta):
    vec_exp = [math.exp(i * beta) for i in vector]
    sum_vec_exp = sum(vec_exp)
    softmax_vec = [i/sum_vec_exp for i in vec_exp]
    return softmax_vec
