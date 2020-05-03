import torch
from torch.autograd import Variable
from SCAN.model import func_attention, cosine_similarity

def cap2img_grounding(img, cap, cap_len, opt):
    imf = img.unsqueeze(0).contiguous()
    n_word = cap_len[0]
    capf = cap[:n_word, :].unsqueeze(0).contiguous()
    weiContext, attn = func_attention(imf, capf, opt, smooth=opt.lambda_softmax)
    cap2img_sim = cosine_similarity(imf, weiContext, dim=2)
    return cap2img_sim

def cands_forward_emb(scan_model, captions, lengths, volatile=False):
    captions = Variable(captions, volatile=volatile)
    if torch.cuda.is_available():
        captions = captions.cuda()
    cap_emb, cap_lens = scan_model.txt_enc(captions, lengths)
    return cap_emb, cap_lens