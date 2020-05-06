import torch


def reduce(num, denom, weights, reduction='micro'):
    if reduction=='micro':
        return torch.sum(num) / torch.sum(denom)
    elif reduction=='macro':
        return torch.mean(num / denom)
    elif reduction=='weighted':
        return torch.sum((num / denom) * (weights / torch.sum(weights)))
    elif reduction is None or reduction=='none':
        return num / denom
    else:
        raise ValueError('Reduction parameter unknown.')
