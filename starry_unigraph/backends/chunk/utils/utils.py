import torch
def select_hot(src, dst, ts, ratio=0.1):
    degree = torch.cat(src,dst,dim=0).bincount()
    _, indices = torch.topk(degree, k=int(ratio * degree.size(0)))
    return indices