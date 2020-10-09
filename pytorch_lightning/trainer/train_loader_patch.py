'''
    This patch solves two problems discussed in 
    https://github.com/PyTorchLightning/pytorch-lightning/pull/1959

    The function  train_dataloader can either return a single instance of 
    torch.utils.data.DataLoader or a dictionary of dataloaders.

    This patch fixes the length and iteration issus 
    and make the rest of the code oblivious of the underlying data structure.

    I will keep the name of the class but a better name is probable advisable

    @christofer-f
'''

import itertools
  
def get_len(d):
    if isinstance(d, dict):
        v = max(d.items(), key=lambda x: len(x[1]))
        return len(v[1])
    else:
        return len(d)

class MagicClass(object):
    def __init__(self, data) -> None:
        super(object, self).__init__()
        self.d = data
        self.l = get_len(data)
 
    def __len__(self) -> int:
        return get_len(self.d)

    def __iter__(self):
        if isinstance(self.d, dict):
            gen = {}
            for k,v in self.d.items():
                gen[k] = itertools.cycle(v)
            for i in range(self.l):
                rv = {}
                for k,v in self.d.items():
                    rv[k] = next(gen[k])
                yield rv
        else:
            gen = itertools.cycle(self.d)
            for i in range(self.l):
                batch = next(gen)
                yield batch