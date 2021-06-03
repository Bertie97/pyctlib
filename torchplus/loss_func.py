import torch

def sigmod_cross_entropy_with_logits(x: torch.Tensor, targets: torch.Tensor, reduction: str="mean"):
    crit = torch.nn.BCEWithLogitsLoss(reduction=reduction)
    return crit(x, targets)

def softmax_cross_entropy_with_logits(x: torch.Tensor, targets: torch.Tensor, indice=-1, reduction: str="mean", keepdim=False):
    """
    Parameter:
    --------
    x: logits
    targets: same dimension as x
        , targets denotes probability distribution
        assert targets.sum(indice, keepdim=keepdim) == 1
    """

    if reduction != "none":
        keepdim = False
    unnorm = - (x * targets).sum(indice, keepdim=keepdim)
    ret = unnorm +  torch.logsumexp(x, indice, keepdim=keepdim)
    if reduction == "mean":
        return ret.mean()
    elif reduction == "sum":
        return ret.sum()
    elif reduction == "none":
        return ret
    else:
        raise ValueError

def softmax_cross_entropy_with_logits_sparse(x: torch.Tensor, target: torch.LongTensor, reduction: str="mean", keepdim=False, smoothing=0):
    if reduction != "none":
        keepdim = False
    if smoothing == 0:
        crit = torch.nn.CrossEntropyLoss(reduction=reduction)
        return crit(x, target)
    else:
        K = x.shape[-1]
        NLL = torch.nn.NLLLoss(reduction="none")
        nll = NLL(x, target) * (1 - K / (K-1) * smoothing)
        ret = nll - smoothing / (K - 1) * x.sum(-1) + torch.logsumexp(x, -1)
        if reduction == "mean":
            return ret.mean()
        elif reduction == "sum":
            return ret.sum()
        elif reduction == "none":
            return ret
        else:
            raise ValueError
