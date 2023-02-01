import logging

from lightning.app import LightningWork

logger = logging.getLogger(__name__)


class AA(LightningWork):
    def __init__(self):
        super().__init__()
        self.message = None

    def run(self):
        self.message = "hello world!"
