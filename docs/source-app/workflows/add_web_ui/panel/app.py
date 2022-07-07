# app.py
import time
import lightning as L
import panel as pn
from panel_frontend import PanelFrontend
from lightning_app.utilities.state import AppState
import datetime as dt

def your_panel_app(lightning_app_state: AppState):
    return pn.Column(
        pn.pane.Markdown("hello"),
        lightning_app_state._plugin.param_state.param.value
    )

class LitPanel(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self._frontend = PanelFrontend(render_fn=your_panel_app)
    
    def configure_layout(self):
        return self._frontend

class LitApp(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.last_update = dt.datetime.now().isoformat()
        self.counter = 0
        self.lit_panel = LitPanel()

    def run(self, *args, **kwargs) -> None:
        time.sleep(0.1)
        if self.counter<2000:
            self.last_update = dt.datetime.now().isoformat()
            self.counter += 1
            print(self.counter, self.last_update)
        return super().run(*args, **kwargs)

    def configure_layout(self):
        tab1 = {"name": "home", "content": self.lit_panel}
        return tab1

app = L.LightningApp(LitApp())