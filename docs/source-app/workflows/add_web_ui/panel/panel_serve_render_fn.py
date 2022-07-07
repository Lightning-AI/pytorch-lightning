"""This file gets run by streamlit, which we launch within Lightning.

From here, we will call the render function that the user provided in ``configure_layout``.
"""
import logging
import os
import pydoc
import sys
from typing import Callable, Union

import panel as pn

from lightning_app.core.flow import LightningFlow
from lightning_app.utilities.state import AppState
from panel_plugin import PanelStatePlugin

logger = logging.getLogger("PanelFrontend")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("starting plugin")
app_state = AppState(plugin=PanelStatePlugin()) # 
logger.info("plugin started")

def _get_render_fn_from_environment() -> Callable:
    render_fn_name = os.environ["LIGHTNING_RENDER_FUNCTION"]
    render_fn_module_file = os.environ["LIGHTNING_RENDER_MODULE_FILE"]
    module = pydoc.importfile(render_fn_module_file)
    return getattr(module, render_fn_name)


def _app_state_to_flow_scope(state: AppState, flow: Union[str, LightningFlow]) -> AppState:
    """Returns a new AppState with the scope reduced to the given flow, as if the given flow as the root."""
    flow_name = flow.name if isinstance(flow, LightningFlow) else flow
    flow_name_parts = flow_name.split(".")[1:]  # exclude root
    flow_state = state
    for part in flow_name_parts:
        flow_state = getattr(flow_state, part)
    return flow_state


def view():
    flow_state = _app_state_to_flow_scope(app_state, flow=os.environ["LIGHTNING_FLOW_NAME"])
    render_fn = _get_render_fn_from_environment()

    return render_fn(app_state)
    
def main():
    logger.info("Panel server starting")
    port=int(os.environ["LIGHTNING_RENDER_PORT"])
    address=os.environ["LIGHTNING_RENDER_ADDRESS"]
    url = os.environ["LIGHTNING_FLOW_NAME"]
    pn.serve({url: view}, address=address, port=port, websocket_origin="*", show=False)
    logger.info("Panel server started on port %s:%s/%s", address, port, url)

# os.environ['LIGHTNING_FLOW_NAME']= 'root.lit_panel'
# os.environ['LIGHTNING_RENDER_FUNCTION']= 'your_panel_app'
# os.environ['LIGHTNING_RENDER_MODULE_FILE']= 'C:\\repos\\private\\lightning\\docs\\source-app\\workflows\\add_web_ui\\panel\\app.py'
# os.environ['LIGHTNING_RENDER_ADDRESS']= 'localhost'
# os.environ['LIGHTNING_RENDER_PORT']= '61965'

main()