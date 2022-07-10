"""This file gets run by Python to lunch a Panel Server with Lightning.

From here, we will call the render_fn that the user provided to the PanelFrontend.

It requires the below environment variables to be set

- LIGHTNING_FLOW_NAME
- LIGHTNING_RENDER_ADDRESS
- LIGHTNING_RENDER_FUNCTION
- LIGHTNING_RENDER_MODULE_FILE
- LIGHTNING_RENDER_PORT

Example:

.. code-block:: bash

        python panel_serve_render_fn
"""
from __future__ import annotations

import logging
import os

import panel as pn

from lightning_app.frontend.utilities.app_state_watcher import AppStateWatcher
from lightning_app.frontend.utilities.other import get_render_fn_from_environment

_logger = logging.getLogger(__name__)


def _view_fn():
    render_fn = get_render_fn_from_environment()
    app = AppStateWatcher()
    return render_fn(app)


def _get_websocket_origin() -> str:
    # Todo: Improve this. I don't know how to find the specific host(s).
    # I tried but it did not work in cloud
    return "*"


def _serve():
    port = int(os.environ["LIGHTNING_RENDER_PORT"])
    address = os.environ["LIGHTNING_RENDER_ADDRESS"]
    url = os.environ["LIGHTNING_FLOW_NAME"]
    websocket_origin = _get_websocket_origin()

    pn.serve({url: _view_fn}, address=address, port=port, websocket_origin=websocket_origin, show=False)
    _logger.debug("Panel server started on port http://%s:%s/%s", address, port, url)


if __name__ == "__main__":
    _serve()
