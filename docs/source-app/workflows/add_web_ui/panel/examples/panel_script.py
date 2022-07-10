import os

import panel as pn

pn.extension(sizing_mode="stretch_width")

pn.panel("# Hello Panel 4").servable()

from app_state_watcher import AppStateWatcher

app = AppStateWatcher()
pn.panel(os.environ.get("PANEL_AUTORELOAD", "no")).servable()
pn.pane.JSON(app.state._state, theme="light", height=300, width=500, depth=3).servable()
