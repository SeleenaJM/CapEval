import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
import csv


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'

        # Captions
        self.captions = []
        with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index/self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader

def ImgCap_Index_Id(opt, data_split):
    dpath = os.path.join(opt.data_path, opt.data_name)
    imgcapId = []
    with open(dpath + '/' + '%s_ids.txt' % data_split, 'rb') as f:
        for line in f:
            imgcapId.append(line.strip())
    return imgcapId

def read_box(box_path):
    box = np.loadtxt(box_path, dtype = 'i', delimiter = ',')
    return box[:,:4]

def get_ref_vocabcaps(opt, vocab):
    dpath = os.path.join(opt.data_path, opt.data_name)
    dset = PrecompDataset(dpath, opt.split_name, vocab)
    captions_sting = dset.captions
    captions_tensor = []

    for i in range(0, len(captions_sting)):
        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(captions_sting[i]).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        captions_tensor.append(target)
    lengths = [len(cap) for cap in captions_tensor]
    targets = torch.zeros(len(captions_tensor), max(lengths)).long()
    for i, cap in enumerate(captions_tensor):
        end = lengths[i]
        targets[i, :end] = cap[:end]  
    return targets  

def get_ref_capstrings(opt):
    dpath = os.path.join(opt.data_path, opt.data_name)
    loc = dpath + '/'
    captions = []
    with open(loc+'%s_caps.txt' % opt.split_name, 'rb') as f:
        for line in f:
            captions.append(line.strip())
    return captions

def read_adversial(vocab, adversial_path, cand_type):
    img_ids = []
    caps_tensor = []
    caps = []
    evals = []
    idxs = []
    count = 0

    with open(adversial_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0 or row[0] == '':
                line_count += 1
                continue
            else:
                if '.jpg' in row[0]:
                    row[0] = row[0].replace('.jpg', '')
                coco_id = row[0]
                if cand_type == 's':
                    coco_eval_sym = '1'
                    coco_cap_sym = row[1]
                    target_sym = cap_preprocessing(coco_cap_sym, coco_eval_sym, vocab)
                    caps_tensor.append(target_sym)
                    img_ids.append(coco_id)
                    caps.append(coco_cap_sym)
                    evals.append(coco_eval_sym)
                    idxs.append(count)

                elif cand_type == 'a':
                    coco_eval_ant = '1'
                    coco_cap_ant = row[2]
                    target_ant = cap_preprocessing(coco_cap_ant, coco_eval_ant, vocab)
                    caps_tensor.append(target_ant)
                    img_ids.append(coco_id)
                    caps.append(coco_cap_ant)
                    evals.append(coco_eval_ant)
                    idxs.append(count)
                count = count + 1

    # Sort a data list by caption length
    data = zip(img_ids, caps_tensor, caps, evals, idxs)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    img_ids, caps_tensor, caps, evals, idxs = zip(*data)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in caps_tensor]
    targets = torch.zeros(len(caps_tensor), max(lengths)).long()
    for i, cap in enumerate(caps_tensor):
        end = lengths[i]
        targets[i, :end] = cap[:end]    

    return img_ids, caps, targets, lengths, evals, idxs




def read_composite_coco(vocab, coco_path, cand_type):
    img_ids = []
    caps_tensor = []
    caps = []
    evals = []
    idxs = []

    with open(coco_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0 or row[0] == '':
                line_count += 1
                continue
            else:
                item = row[0].split('_')
                coco_id = item[len(item)-1].replace('.jpg', '').lstrip('0')
                if cand_type == 'h':
                    coco_cap1 = row[1]
                    coco_eval1 = row[4]
                    target1 = cap_preprocessing(coco_cap1, coco_eval1, vocab)
                    caps_tensor.append(target1)
                    img_ids.append(coco_id)
                    caps.append(coco_cap1)
                    evals.append(coco_eval1)
                    idxs.append(caps.index(coco_cap1))

                elif cand_type == 'm1':
                    coco_cap2 = row[2]
                    coco_eval2 = row[5]
                    target2 = cap_preprocessing(coco_cap2, coco_eval2, vocab)
                    caps_tensor.append(target2)
                    img_ids.append(coco_id)
                    caps.append(coco_cap2)
                    evals.append(coco_eval2)
                    idxs.append(caps.index(coco_cap2))

                elif cand_type == 'm2':
                    coco_cap3 = row[3]
                    coco_eval3 = row[6]
                    target3 = cap_preprocessing(coco_cap3, coco_eval3, vocab)
                    caps_tensor.append(target3)
                    img_ids.append(coco_id)
                    caps.append(coco_cap3)
                    evals.append(coco_eval3)
                    idxs.append(caps.index(coco_cap3))
    
    # Sort a data list by caption length
    data = zip(img_ids, caps_tensor, caps, evals, idxs)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    img_ids, caps_tensor, caps, evals, idxs = zip(*data)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in caps_tensor]
    targets = torch.zeros(len(caps_tensor), max(lengths)).long()
    for i, cap in enumerate(caps_tensor):
        end = lengths[i]
        targets[i, :end] = cap[:end]    

    return img_ids, caps, targets, lengths, evals, idxs

def cap_preprocessing(coco_cap, coco_eval, vocab):
     # Convert caption (string) to word ids.
    tokens = nltk.tokenize.word_tokenize(
        str(coco_cap).lower().decode('utf-8'))
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    target = torch.Tensor(caption)
    return target


def get_overlap_annotation(opt, vocab):
    imgcap_id = ImgCap_Index_Id(opt, opt.split_name)
    comp_ids, comp_caps, comp_evals = read_composite_coco('/home/seleenaj/Documents/CV_Data/Composite/experiment/30k_correctness.csv')

    overlap_ids_idx = {}
    for i in comp_ids:
        if i in imgcap_id:
            overlap_ids_idx[comp_ids.index(i)] = imgcap_id.index(i)

    
    overlap_caps_idx = {}
    dpath = os.path.join(opt.data_path, opt.data_name)
    dset = PrecompDataset(dpath, opt.split_name, vocab)

    for ikey in overlap_ids_idx.keys():

        star_idx = overlap_ids_idx[ikey]
        end_idx = star_idx + 5
        temp_caps = []
        for l in range(star_idx, end_idx):
            image, target, index, img_id = dset[l]
            word_vocab_idxs = target.tolist()
            words = [vocab.idx2word[str(int(i))] for i in word_vocab_idxs][1:-1]
            temp_caps.append(' '.join(words))
            capstring1 = ' '.join(words[0:5])
            capstring2 = ' '.join(words[len(words)-5:len(words)])
            if capstring1 in comp_caps[ikey].lower() or capstring2 in comp_caps[ikey].lower():
                overlap_caps_idx[ikey] = l
        if ikey not in overlap_caps_idx:
            print(temp_caps)
            print('\n')
            print(comp_caps[ikey].lower())
            print('\n')
            temp_caps = []

    return overlap_caps_idx, comp_evals

def get_overlap_items(imgcap_id, comp_ids):
    overlap_ids_idx = {}
    for i in comp_ids:
        if i in imgcap_id:
            overlap_ids_idx[comp_ids.index(i)] = imgcap_id.index(i)

    return overlap_ids_idx