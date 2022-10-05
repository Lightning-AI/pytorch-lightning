from typing import Callable

from lightning import LightningApp, LightningFlow
from lightning.app.frontend import JustPyFrontend


class Flow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def run(self):
        print(self.counter)

    def configure_layout(self):
        return JustPyFrontend(render_fn=render_fn)


def render_fn(get_state: Callable) -> Callable:
    import justpy as jp

    def my_click(self, *_):
        state = get_state()
        old_counter = state.counter
        state.counter += 1
        self.text = f"Click Me ! Old Counter: {old_counter} New Counter: {state.counter}"

    def website():
        wp = jp.WebPage()
        d = jp.Div(text="Hello ! Click Me!")
        d.on("click", my_click)
        wp.add(d)
        return wp

    return website


app = LightningApp(Flow())
