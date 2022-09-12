import lightning as L


class EmptyWork(L.LightningWork):
    def run(self):
        print("HERE")


class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.work_A = EmptyWork(parallel=True)
        self.work_B = EmptyWork(parallel=True)

    def run(self):
        self.work_A.run()
        self.work_B.run()


app = L.LightningApp(RootFlow())
