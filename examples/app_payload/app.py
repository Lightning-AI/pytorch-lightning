import lightning as L
from lightning.app.storage.payload import Payload


class SourceFileWriterWork(L.LightningWork):
    def __init__(self):
        super().__init__()
        self.value = None

    def run(self):
        self.value = Payload(42)


class DestinationWork(L.LightningWork):
    def run(self, payload):
        assert payload.value == 42


class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.src = SourceFileWriterWork()
        self.dst = DestinationWork()

    def run(self):
        self.src.run()
        self.dst.run(self.src.value)
        self._exit("Application End!")


app = L.LightningApp(RootFlow())
