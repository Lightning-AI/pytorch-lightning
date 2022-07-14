from lightning_app import LightningApp, LightningFlow, LightningWork
from lightning_app.utilities.app_helpers import pretty_state


class Work(LightningWork):
    def __init__(self):
        super().__init__(cache_calls=False)
        # Attributes are registered automatically in the state.
        self.counter = 0

    def run(self):
        # Incrementing an attribute gets reflected in the `RootFlow` state.
        self.counter += 1


class Flow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.w = Work()

    def run(self):
        if self.w.has_started:
            print(f"State: {pretty_state(self.state)} \n")
        self.w.run()


app = LightningApp(Flow())
