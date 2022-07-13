# app.py
import datetime as dt

import panel as pn
from app_state_watcher import AppStateWatcher

# Todo: change import
from panel_frontend import PanelFrontend

import lightning as L

pn.extension(sizing_mode="stretch_width")


def your_panel_app(app: AppStateWatcher):
    @pn.depends(app.param.state)
    def last_update(_):
        return f"last_update: {app.state.last_update}"

    return pn.Column(
        last_update,
    )


class LitPanel(L.LightningFlow):
    def __init__(self):
        super().__init__()

        self._frontend = PanelFrontend(your_panel_app)
        self._last_update = dt.datetime.now()
        self.last_update = self._last_update.isoformat()

    def run(self):
        now = dt.datetime.now()
        if (now - self._last_update).microseconds > 200:
            self._last_update = now
            self.last_update = self._last_update.isoformat()

    def configure_layout(self):
        return self._frontend


class LitApp(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.lit_panel = LitPanel()

    def run(self) -> None:
        self.lit_panel.run()

    def configure_layout(self):
        return {"name": "home", "content": self.lit_panel}


app = L.LightningApp(LitApp())
