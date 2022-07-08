"""This file gets run by streamlit, which we launch within Lightning.

From here, we will call the render function that the user provided in ``configure_layout``.
"""
from __future__ import annotations
import logging
import os
import sys

import panel as pn

from lightning_app.frontend.streamlit_base import _get_render_fn_from_environment
from panel_utils import AppStateWatcher

logger = logging.getLogger("PanelFrontend")

logger.setLevel(logging.DEBUG)

logger.debug("starting plugin")
logger.debug("plugin started")

app_state_watcher: None | AppStateWatcher = None

def main():
    def view():
        global app_state_watcher
        if not app_state_watcher:
            app_state_watcher = AppStateWatcher()
        render_fn = _get_render_fn_from_environment()
        return render_fn(app_state_watcher)
    
    logger.debug("Panel server starting")
    port=int(os.environ["LIGHTNING_RENDER_PORT"])
    address=os.environ["LIGHTNING_RENDER_ADDRESS"]
    url = os.environ["LIGHTNING_FLOW_NAME"]
    pn.serve({url: view}, address=address, port=port, websocket_origin="*", show=False)
    logger.debug("Panel server started on port %s:%s/%s", address, port, url)

# os.environ['LIGHTNING_FLOW_NAME']= 'root.lit_panel'
# os.environ['LIGHTNING_RENDER_FUNCTION']= 'your_panel_app'
# os.environ['LIGHTNING_RENDER_MODULE_FILE']= 'C:\\repos\\private\\lightning\\docs\\source-app\\workflows\\add_web_ui\\panel\\app.py'
# os.environ['LIGHTNING_RENDER_ADDRESS']= 'localhost'
# os.environ['LIGHTNING_RENDER_PORT']= '61965'

main()