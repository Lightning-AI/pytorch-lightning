import logging
import os
import pydoc
import sys
from typing import Callable, Union

import panel as pn

from lightning_app.core.flow import LightningFlow
from lightning_app.utilities.state import AppState
from panel_plugin import PanelStatePlugin


def test_param_state():
    app_state = AppState(plugin=PanelStatePlugin()) #
    assert hasattr(app_state._plugin.param_state) 
