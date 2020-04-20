import torch

from pytorch_lightning.utilities import rank_zero_warn

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True


# Utility function to copy data to given device
# Works for any form of nested lists, tuples or dictionaries containting tensors
def transfer_data_to_device(batch, device_type, idx=None, warn_on_transfer=False):
    # Deal with TPUs separately, they don't use device indexes for some reason
    if device_type == 'tpu' and XLA_AVAILABLE:
        if callable(getattr(batch, 'to', None)):
            if warn_on_transfer:
                rank_zero_warn('Auto transferred data to device {}'.format(xm.xla_device()))
            return batch.to(xm.xla_device())

    # base case: nothing to do
    device = torch.device(device_type, idx)
    if torch.is_tensor(batch) and batch.device == device:
        return batch

    # object can be directly moved using `cuda` or `to`
    if callable(getattr(batch, 'cuda', None)) and device_type == 'cuda':
        if warn_on_transfer:
            rank_zero_warn('Auto transferred data to device {}'.format(device))
        return batch.cuda(device=device)

    if callable(getattr(batch, 'to', None)):
        if warn_on_transfer:
            rank_zero_warn('Auto transferred data to device {}'.format(device))
        return batch.to(device=device)

    # when list or tuple
    if isinstance(batch, (list, tuple)):
        if isinstance(batch, tuple):
            batch = list(batch)
        for i, x in enumerate(batch):
            batch[i] = transfer_data_to_device(x, device_type, idx, warn_on_transfer)
        return batch

    # when dict
    if isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = transfer_data_to_device(v, device_type, idx, warn_on_transfer)
        return batch

    # nothing matches, return the value as is without transform
    return batch
