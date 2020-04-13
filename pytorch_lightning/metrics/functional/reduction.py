import torch


def reduce(to_reduce: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    reduces a given tensor by a given reduction method
    Parameters
    ----------
    to_reduce : torch.Tensor
        the tensor, which shall be reduced
    reduction : str
        a string specifying the reduction method.
        should be one of 'elementwise_mean' | 'none' | 'sum'
    Returns
    -------
    torch.Tensor
        reduced Tensor
    Raises
    ------
    ValueError
        if an invalid reduction parameter was given
    """
    if reduction == 'elementwise_mean':
        return torch.mean(to_reduce)
    if reduction == 'none':
        return to_reduce
    if reduction == 'sum':
        return torch.sum(to_reduce)
    raise ValueError('Reduction parameter unknown.')
