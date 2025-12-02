import torch

def compute_topk_accuracy(output, target, topk=(1,5)):
    """Top-k 정확도 계산"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True) # [B, maxk]
    pred = pred.t() # [maxk, B]
    correct = pred.eq(target.view(1,-1).expand_as(pred)) # [maxk, B]

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        acc = correct_k * 100.0 / batch_size
        res.append(acc.item())
    return res