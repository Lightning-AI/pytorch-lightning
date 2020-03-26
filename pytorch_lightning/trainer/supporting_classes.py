import torch


class TensorRunningMean(object):
    """
    Tracks a running mean without graph references.
    Round robbin for the mean
    """
    def __init__(self, window_length):
        self.window_length = window_length
        self.reset()

    def reset(self):
        self.memory = torch.Tensor(self.window_length)
        self.current_idx = 0

    def append(self, x):
        # map proper type for memory if they don't match
        if self.memory.type() != x.type():
            self.memory.type_as(x)

        # store without grads
        with torch.no_grad():
            self.memory[self.current_idx] = x

        # increase index
        self.current_idx += 1

        # reset index when hit limit of tensor
        if self.current_idx >= self.window_length:
            self.current_idx = 0

    def mean(self):
        return self.memory.mean()
