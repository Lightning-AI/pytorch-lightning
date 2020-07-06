from pytorch_lightning.core import LightningModule
from pytorch_lightning.core.decorators import auto_move_data


def infer(network: LightningModule, input, infer_mode='eval'):
    INFER_MODE = {
        'eval': network.freeze,
        'train': network.unfreeze
    }
    network.forward = auto_move_data(network.forward)
    INFER_MODE[infer_mode]()

    return network(input)
