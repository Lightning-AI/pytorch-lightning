from pytorch_lightning.core import LightningModule


def infer(network: LightningModule, x, infer_mode='eval'):
    INFER_MODE = {
        'eval': network.freeze,
        'train': network.unfreeze
    }
    INFER_MODE[infer_mode]()

    return network(x)
