import torch

def sumofsquare(x):
    return torch.sum(torch.square(x))

def sigmod_cross_entropy_with_logits(x: torch.Tensor, targets: torch.Tensor, reduction: str="mean") -> torch.Tensor:
    crit = torch.nn.BCEWithLogitsLoss(reduction=reduction)
    return crit(x, targets)

def softmax_cross_entropy_with_logits(x: torch.Tensor, targets: torch.Tensor, indice: int=-1, reduction: str="mean", keepdim: bool=False, target_sum_one=True) -> torch.Tensor:
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
    if not target_sum_one:
        x = torch.softmax(x, dim=indice)
        ret = - (targets * torch.log(x)).sum(indice, keepdim=keepdim)
    else:
        unnorm = - (x * targets).sum(indice, keepdim=keepdim)
        ret = unnorm + torch.logsumexp(x, indice, keepdim=keepdim)
    if reduction == "mean":
        return ret.mean()
    elif reduction == "sum":
        return ret.sum()
    elif reduction == "none":
        return ret
    else:
        raise ValueError

def softmax_cross_entropy_with_logits_sparse(x: torch.Tensor, target: torch.LongTensor, class_index=-1, reduction: str="mean", keepdim: bool=False, smoothing: float=0, mask=None) -> torch.Tensor:
    if reduction != "none":
        keepdim = False
    class_index = class_index % len(x.shape)
    assert (*x.shape[:class_index], *x.shape[class_index+1:]) == target.shape
    if smoothing == 0:
        negative_logsoftmax = - torch.nn.LogSoftmax(class_index)(x)
        if mask is not None:
            new_target = torch.where(mask, target, 0)
            ret = torch.gather(negative_logsoftmax, class_index, new_target.unsqueeze(class_index)).squeeze(class_index)
            ret = torch.where(mask, ret, torch.zeros_like(ret))
            return ret
        else:
            ret = torch.gather(negative_logsoftmax, class_index, target.unsqueeze(class_index)).squeeze(class_index)
            if reduction == "mean":
                return ret.mean()
            elif reduction == "sum":
                return ret.sum()
            elif reduction == "none":
                return ret
            raise ValueError()
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

def entropy(x: torch.Tensor, normalized: bool=True, dim: int=-1, reduction: str="none") -> torch.Tensor:
    if not normalized:
        x = x.softmax(dim)
    entropy = - (x * torch.log(x + 1e-30)).sum(dim=dim)
    if reduction == "none":
        return entropy
    elif reduction == "mean":
        return entropy.mean()
    elif reduction == "sum":
        return entropy.sum()
    raise ValueError

def kl_divergence_with_logits(x: torch.Tensor, target: torch.Tensor, reduction: str="mean") -> torch.Tensor:
    """
    x: shape[..., n], unnormalized log of probability of Q
    target: same shape as x, probability of P
    KL(p, q) = p log(p/q) = target log(target / softmax(x))
    """
    ret = torch.nn.functional.kl_div(x - torch.logsumexp(x, dim=-1, keepdim=True), target, reduction="none")
    if reduction == "mean":
        return ret.sum(-1).mean()
    elif reduction == "sum":
        return ret.sum()
    elif reduction == "none":
        return ret.sum(-1)
    raise ValueError
