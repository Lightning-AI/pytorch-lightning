from lightning.app.core.app import LightningApp
from lightning.app.core.flow import LightningFlow
from lightning.app.core.work import LightningWork
from lightning.app.frontend.web import StaticWebFrontend
from lightning.app.utilities.packaging.cloud_compute import CloudCompute


class WorkA(LightningWork):
    def __init__(self):
        """WorkA."""
        super().__init__()

    def run(self):
        pass


class WorkB(LightningWork):
    def __init__(self):
        """WorkB."""
        super().__init__(cloud_compute=CloudCompute("gpu"))

    def run(self):
        pass


class FlowA(LightningFlow):
    def __init__(self):
        """FlowA Component."""
        super().__init__()
        self.work_a = WorkA()

    def run(self):
        pass


class FlowB(LightningFlow):
    def __init__(self):
        """FlowB."""
        super().__init__()
        self.work_b = WorkB()

    def run(self):
        pass

    def configure_layout(self):
        return StaticWebFrontend(serve_dir=".")


class RootFlow(LightningFlow):
    def __init__(self):
        """RootFlow."""
        super().__init__()
        self.flow_a_1 = FlowA()
        self.flow_a_2 = FlowA()
        self.flow_b = FlowB()

    def run(self):
        self.stop()


app = LightningApp(RootFlow())
