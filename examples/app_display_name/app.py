import lightning as L


class Work(L.LightningWork):
    def __init__(self):
        super().__init__(start_with_flow=False)

    def run(self):
        pass


class Flow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.w = Work()
        self.w.display_name = "My Custom Name"

    def run(self):
        self.w.run()


app = L.LightningApp(Flow())
