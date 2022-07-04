"""This app demonstrates how the PanelFrontend can read/ pull the global app
state and display it"""
import datetime

import lightning as L
import panel as pn
from panel_frontend import PanelFrontend
from pathlib import Path
import json

DATETIME_FORMAT = "%Y-%m-%d, %H:%M:%S.%f"

APP_STATE = Path(__file__).parent / "app_state.json"

def read_app_state():
    with open(APP_STATE) as json_file:
        return json.load(json_file)

def save_app_state(state):
    with open(APP_STATE, 'w') as outfile:
        json.dump(state, outfile)

def to_string(value: datetime.datetime)->str:
    return value.strftime(DATETIME_FORMAT)

def render_fn(self):
    global_app_state_pane = pn.pane.JSON(depth=2)
    last_local_update_pane = pn.pane.Str()
    
    def update():
        last_local_update_pane.object= "last local update: " + to_string(datetime.datetime.now())
        
        # Todo: Figure out the right way to read/ pull the app state
        global_app_state_pane.object = read_app_state()

    # Todo: Figure out how to schedule the callback globally to make the app scale
    # There is no reason that each session should read or pull into memory individually
    pn.state.add_periodic_callback(update, period=1000)
    # Todo: Refactor the Panel app implementation to a more reactive api
    # Todo: Giver the Panel app a nicer UX
    return pn.Column(last_local_update_pane, global_app_state_pane)

class Flow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        
        self.panel_frontend = PanelFrontend(render_fn=render_fn)
        self._last_global_update = datetime.datetime.now()
        self.last_global_update = to_string(self._last_global_update)

    def run(self):
        self.panel_frontend.run()
        now = datetime.datetime.now()
        if (now-self._last_global_update).microseconds>=100:
            save_app_state(self.state)
            self._last_global_update=now
            self.last_global_update = to_string(now)

    def configure_layout(self):
        tab1 = {"name": "Home", "content": self.panel_frontend}
        return tab1

app = L.LightningApp(Flow())

if __name__.startswith("bokeh"):
    render_fn(None).servable()