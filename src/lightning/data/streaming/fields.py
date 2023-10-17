from dataclasses import dataclass

from torch import Tensor


@dataclass
class Data:
    pass


class Text(Data):
    tokens: Tensor
