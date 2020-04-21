def recursive_detach(in_dict: dict) -> dict:
    """Detach all tensors in `in_dict`.

    May operate recursively if some of the values in `in_dict` are dictionaries
    which contain instances of `torch.Tensor`. Other types in `in_dict` are
    not affected by this utility function.

    Args:
        in_dict:

    Return:
        out_dict:
    """
    out_dict = {}
    for k, v in in_dict.items():
        if isinstance(v, dict):
            out_dict.update({k: recursive_detach(v)})
        elif callable(getattr(v, 'detach', None)):
            out_dict.update({k: v.detach()})
        else:
            out_dict.update({k: v})
    return out_dict
