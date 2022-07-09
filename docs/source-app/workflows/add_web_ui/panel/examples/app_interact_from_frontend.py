# app.py
import panel as pn

import lightning as L
# Todo: Change import
from panel_frontend import PanelFrontend
from app_state_watcher import AppStateWatcher
pn.extension(sizing_mode="stretch_width")

def your_panel_app(app: AppStateWatcher):

    submit_button = pn.widgets.Button(name="submit")

    @pn.depends(submit_button, watch=True)
    def submit(_):
        app.state.count += 1

    @pn.depends(app.param.state)
    def current_count(_):
        return f"current count: {app.state.count}"

    return pn.Column(
        submit_button,
        current_count,
    )


class LitPanel(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self._frontend = PanelFrontend(render_fn=your_panel_app)
        self.count = 0

    def configure_layout(self):
        return self._frontend


class LitApp(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.lit_panel = LitPanel()

    def configure_layout(self):
        return {"name": "home", "content": self.lit_panel}


app = L.LightningApp(LitApp())
