import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import numpy as np
import csv

def cap2imgid(opt, data_split):
    dpath = os.path.join(opt.data_path, opt.data_name)
    imgcapId = []
    with open(dpath + '/' + '%s_ids.txt' % data_split, 'rb') as f:
        for line in f:
            imgcapId.append(line.strip())
    return imgcapId

def cap_preprocessing(cap, vocab):
     # Convert caption (string) to word ids.
    tokens = nltk.tokenize.word_tokenize(
        str(cap).lower().decode('utf-8'))
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    target = torch.Tensor(caption)
    return target

def get_ref_capstrings(opt):
    dpath = os.path.join(opt.data_path, opt.data_name)
    loc = dpath + '/'
    captions = []
    with open(loc+'%s_caps.txt' % opt.split_name, 'rb') as f:
        for line in f:
            captions.append(line.strip())
    return captions

def flickr8k_imgname2idx(data_path, data_name, split_name):
    ref_imgids = []
    ref_imgnames = []
    with(open(os.path.join(data_path, data_name, split_name)+'_ids.txt')) as f1:
        for line in f1:
            imgid = line.replace('\n', '')
            ref_imgids.append(imgid)
    with(open(os.path.join(data_path, data_name, split_name)+'_names.txt')) as f2:
        for line in f2:
            imgname = line.replace('\n', '')
            ref_imgnames.append(imgname)
    return ref_imgids, ref_imgnames


def read_composite(vocab, opt, source):
    imgids = []
    caps = {'h':{}, 'm1':{}, 'm2':{}}
    for sys in caps.keys():
        caps[sys]['tensor'] = []
        caps[sys]['string'] = []
        caps[sys]['eval'] = []

    if source == '8k':
        ref_imgids, ref_imgnames = flickr8k_imgname2idx(opt.data_path, opt.data_name, opt.split_name)
    
    loc = opt.candidate_path+'/composite/'+source+'_correctness.csv'
    with open(loc) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0 or row[0] == '':
                line_count += 1
                continue
            else:
                line_count += 1
                if source == 'coco':
                    item = row[0].split('_')
                elif source == '30k' or source == '8k':
                    item = row[0].split('/')
                
                imgid = item[len(item)-1].replace('.jpg', '').lstrip('0')
                
                imgids.append(imgid)
                for k in range(1,4):
                    target = cap_preprocessing(row[k], vocab)
                    if k == 1:
                        sys = 'h'
                    elif k == 2:
                        sys = 'm1'
                    elif k == 3:
                        sys = 'm2'
                    caps[sys]['tensor'].append(target)
                    caps[sys]['string'].append(row[k])
                    caps[sys]['eval'].append(row[k+3])        
    for sys,val in caps.items():
        sys_imgids = imgids
        # Sort a data list by caption length
        data = zip(sys_imgids, val['tensor'], val['string'], val['eval'])
        data.sort(key=lambda x: len(x[1]), reverse=True)
        sys_imgids, sys_caps_tensor, sys_caps_string, sys_caps_evals = zip(*data)
        # Merget captions (convert tuple of 1D tensor to 2D tensor)
        caps_lengths = [len(cap) for cap in sys_caps_tensor]
        targets = torch.zeros(len(sys_caps_tensor), max(caps_lengths)).long()
        for i, cap in enumerate(sys_caps_tensor):
            end = caps_lengths[i]
            targets[i, :end] = cap[:end]

        caps[sys]['imgid'] = sys_imgids
        caps[sys]['tensor'] = targets
        caps[sys]['string'] = sys_caps_string
        caps[sys]['length'] = caps_lengths
        caps[sys]['eval'] = sys_caps_evals 
    return caps


def read_usecase(vocab, opt):
    imgids = []
    caps_tensor = []
    caps_string = []
    with open(os.path.join(opt.candidate_path, 'usecase', 'output_cap.txt')) as f:
        for line in f:
            item = line.strip().split('\t')
            imgid = item[0]
            cstr = item[1]
            target = cap_preprocessing(cstr, vocab)
            
            imgids.append(imgid)
            caps_tensor.append(target)
            caps_string.append(cstr)
    # Sort a data list by caption length
    data = zip(imgids, caps_tensor, caps_string)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    imgids, caps_tensor, caps_string = zip(*data)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    caps_lengths = [len(cap) for cap in caps_tensor]
    targets = torch.zeros(len(caps_tensor), max(caps_lengths)).long()
    for i, cap in enumerate(caps_tensor):
        end = caps_lengths[i]
        targets[i, :end] = cap[:end]
    return imgids, caps_string, targets, caps_lengths



def read_flickr8k(vocab, opt):
    imgids = []
    caps_id = []
    caps_tensor = []
    caps_string = []
    caps_evals = []
    caps_idxs = []

    all_caps = {} 
    with open(os.path.join(opt.candidate_path, 'flickr8k', 'Flickr8k.token.txt'), 'rb') as f:
        for line in f:
            item = line.split('\t')
            all_caps[item[0]] = item[1].replace('\n', '')

    with open(os.path.join(opt.candidate_path, 'flickr8k', 'ExpertAnnotations.txt')) as f:
        for line in f:
            item = line.split('\t')
            imgid = item[0]
            cid = item[1]

            ##remove candidates that are actually belonged to the target image
            if(cid.split('#')[0] == imgid):
                continue

            cstr = all_caps[cid]
            ceva = []
            ceva.append(item[2])
            ceva.append(item[3])
            ceva.append(item[4].replace('\n', ''))
            target = cap_preprocessing(cstr, vocab)

            #imgids.append(sudoid)
            imgids.append(imgid)
            caps_id.append(cid)
            caps_tensor.append(target)
            caps_string.append(cstr)
            caps_evals.append(ceva)
            caps_idxs.append(caps_string.index(cstr))

    # Sort a data list by caption length
    data = zip(imgids, caps_tensor, caps_string, caps_evals, caps_id, caps_idxs)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    imgids, caps_tensor, caps_string, caps_evals, caps_id, caps_idxs = zip(*data)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    caps_lengths = [len(cap) for cap in caps_tensor]
    targets = torch.zeros(len(caps_tensor), max(caps_lengths)).long()
    for i, cap in enumerate(caps_tensor):
        end = caps_lengths[i]
        targets[i, :end] = cap[:end]    
    return imgids, caps_string, targets, caps_lengths, caps_evals, caps_id, caps_idxs

def read_pascal(vocab, opt):
    imgids = []
    caps_tensor = []
    caps_string = []
    caps_type = []
    caps_idxs = []
    caps_pairevals = []
    caps_pairtype = []
    caps_pairid = []

    with open(os.path.join(opt.candidate_path, 'pascal', 'pascal_test.txt')) as f:
        for line in f:
            item = line.replace('\n', '').split('\t')

            for k in range(2,4):
                if k == 2:
                    cstr = item[k]
                    ctype = 'B'
                if k == 3:
                    cstr = item[k]
                    ctype = 'C'

                imgids.append(item[1])
                target = cap_preprocessing(cstr, vocab)
                caps_tensor.append(target)
                caps_string.append(cstr)
                caps_type.append(ctype)
                caps_idxs.append(caps_string.index(cstr))
                caps_pairid.append(int(item[0]))
                caps_pairevals.append(item[4])
                caps_pairtype.append(item[5])

    # Sort a data list by caption length
    data = zip(imgids, caps_tensor, caps_string, caps_type, caps_idxs, caps_pairevals, caps_pairtype, caps_pairid)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    imgids, caps_tensor, caps_string, caps_type, caps_idxs, caps_pairevals, caps_pairtype, caps_pairid = zip(*data)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    caps_lengths = [len(cap) for cap in caps_tensor]
    targets = torch.zeros(len(caps_tensor), max(caps_lengths)).long()
    for i, cap in enumerate(caps_tensor):
        end = caps_lengths[i]
        targets[i, :end] = cap[:end]
    return imgids, caps_string, targets, caps_lengths, caps_type, caps_idxs, caps_pairevals, caps_pairtype, caps_pairid