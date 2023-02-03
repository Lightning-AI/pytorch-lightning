from lightning.app import LightningApp, LightningFlow


class EmptyFlow(LightningFlow):
    def run(self):
        pass


if __name__ == "__main__":
    # trigger a Python exception `IndexError: list index out of range` before we can load the app
    _ = [1, 2, 3][4]

    app = LightningApp(EmptyFlow())
