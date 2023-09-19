from lightning.app import LightningApp, LightningFlow, LightningWork
from lightning.app.storage import Payload


class SourceFileWriterWork(LightningWork):
    def __init__(self):
        super().__init__()
        self.value = None

    def run(self):
        self.value = Payload(42)


class DestinationWork(LightningWork):
    def run(self, payload):
        assert payload.value == 42


class RootFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.src = SourceFileWriterWork()
        self.dst = DestinationWork()

    def run(self):
        self.src.run()
        self.dst.run(self.src.value)
        self.stop("Application End!")


app = LightningApp(RootFlow())
