from lightning.app import LightningApp, LightningFlow


class EmptyFlow(LightningFlow):
    def run(self):
        pass


app_1 = LightningApp(EmptyFlow())
app_2 = LightningApp(EmptyFlow())
