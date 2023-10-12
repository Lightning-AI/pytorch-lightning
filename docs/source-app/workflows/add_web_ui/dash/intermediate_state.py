import dash
import dash_daq as daq
import dash_renderjson
from dash import html, Input, Output

from lightning.app import LightningWork, LightningFlow, LightningApp
from lightning.app.utilities.state import AppState


class LitDash(LightningWork):
    def run(self):
        dash_app = dash.Dash(__name__)

        dash_app.layout = html.Div([daq.ToggleSwitch(id="my-toggle-switch", value=False), html.Div(id="output")])

        @dash_app.callback(Output("output", "children"), [Input("my-toggle-switch", "value")])
        def display_output(value):
            if value:
                state = AppState()
                state._request_state()
                return dash_renderjson.DashRenderjson(id="input", data=state._state, max_depth=-1, invert_theme=True)

        dash_app.run_server(host=self.host, port=self.port)


class LitApp(LightningFlow):
    def __init__(self):
        super().__init__()
        self.lit_dash = LitDash(parallel=True)

    def run(self):
        self.lit_dash.run()

    def configure_layout(self):
        tab1 = {"name": "home", "content": self.lit_dash}
        return tab1


app = LightningApp(LitApp())
