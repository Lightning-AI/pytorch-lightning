"""This file gets run by Python to lunch a Panel Server with Lightning.

From here, we will call the render_fn that the user provided to the PanelFrontend.

It requires the below environment variables to be set

- LIGHTNING_RENDER_FUNCTION
- LIGHTNING_RENDER_MODULE_FILE

Example:

.. code-block:: bash

        python panel_serve_render_fn
"""
import inspect
import os

import panel as pn

from lightning_app.frontend.utilities.app_state_watcher import AppStateWatcher
from lightning_app.frontend.utilities.other import get_render_fn_from_environment


def _get_render_fn():
    render_fn_name = os.environ["LIGHTNING_RENDER_FUNCTION"]
    render_fn_module_file = os.environ["LIGHTNING_RENDER_MODULE_FILE"]
    render_fn = get_render_fn_from_environment(render_fn_name, render_fn_module_file)
    if inspect.signature(render_fn).parameters:

        def _render_fn_wrapper():
            app = AppStateWatcher()
            return render_fn(app)

        return _render_fn_wrapper
    return render_fn


if __name__.startswith("bokeh"):
    # I use caching for efficiency reasons. It shaves off 10ms from having
    # to get_render_fn_from_environment every time
    if not "lightning_render_fn" in pn.state.cache:
        pn.state.cache["lightning_render_fn"] = _get_render_fn()
    pn.state.cache["lightning_render_fn"]()
