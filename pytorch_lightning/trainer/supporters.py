import torch


class TensorRunningMean(object):
    """
    Tracks a running mean without graph references.
    Round robbin for the mean

    Examples:
        >>> accum = TensorRunningMean(5)
        >>> accum.last(), accum.mean()
        (None, None)
        >>> accum.append(torch.tensor(1.5))
        >>> accum.last(), accum.mean()
        (tensor(1.5000), tensor(1.5000))
        >>> accum.append(torch.tensor(2.5))
        >>> accum.last(), accum.mean()
        (tensor(2.5000), tensor(2.))
        >>> accum.reset()
        >>> _= [accum.append(torch.tensor(i)) for i in range(13)]
        >>> accum.last(), accum.mean()
        (tensor(12.), tensor(10.))
    """
    def __init__(self, window_length: int):
        self.window_length = window_length
        self.memory = torch.Tensor(self.window_length)
        self.current_idx: int = 0
        self.last_idx: int = None
        self.rotated: bool = False

    def reset(self) -> None:
        self = TensorRunningMean(self.window_length)

    def last(self):
        if self.last_idx is not None:
            return self.memory[self.last_idx]

    def append(self, x):
        # map proper type for memory if they don't match
        if self.memory.type() != x.type():
            self.memory.type_as(x)

        # store without grads
        with torch.no_grad():
            self.memory[self.current_idx] = x
            self.last_idx = self.current_idx

        # increase index
        self.current_idx += 1

        # reset index when hit limit of tensor
        self.current_idx = self.current_idx % self.window_length
        if self.current_idx == 0:
            self.rotated = True

    def mean(self):
        if self.last_idx is not None:
            return self.memory.mean() if self.rotated else self.memory[:self.current_idx].mean()
