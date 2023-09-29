# v0_app.py
import os
from datetime import datetime
from time import sleep

from lightning.app import LightningApp, LightningFlow
from lightning.app.frontend import StaticWebFrontend


class Word(LightningFlow):
    def __init__(self, letter):
        super().__init__()
        self.letter = letter
        self.repeats = letter

    def run(self):
        self.repeats += self.letter

    def configure_layout(self):
        return StaticWebFrontend(os.path.join(os.path.dirname(__file__), f"ui/{self.letter}"))


class V0App(LightningFlow):
    def __init__(self):
        super().__init__()
        self.aas = Word("a")
        self.bbs = Word("b")
        self.counter = 0

    def run(self):
        now = datetime.now()
        now = now.strftime("%H:%M:%S")
        log = {"time": now, "a": self.aas.repeats, "b": self.bbs.repeats}
        print(log)
        self.aas.run()
        self.bbs.run()

        sleep(2.0)
        self.counter += 1

    def configure_layout(self):
        tab1 = {"name": "Tab_1", "content": self.aas}
        tab2 = {"name": "Tab_2", "content": self.bbs}
        tab3 = {"name": "Tab_3", "content": "https://tensorboard.dev/experiment/8m1aX0gcQ7aEmH0J7kbBtg/#scalars"}

        return [tab1, tab2, tab3]


app = LightningApp(V0App(), log_level="debug")
