from lightning.app import LightningApp, LightningFlow, LightningWork


class Work(LightningWork):
    def __init__(self, start_with_flow=True):
        super().__init__(start_with_flow=start_with_flow)

    def run(self):
        pass


class Flow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.w = Work()
        self.w1 = Work(start_with_flow=False)
        self.w.display_name = "My Custom Name"  # Not supported yet
        self.w1.display_name = "My Custom Name 1"

    def run(self):
        self.w.run()
        self.w1.run()


app = LightningApp(Flow())
