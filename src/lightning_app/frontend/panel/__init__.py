"""The PanelFrontend and AppStateWatcher makes it easy to create lightning apps with the Panel data app
framework."""
from lightning_app.frontend.panel.panel_frontend import PanelFrontend
from lightning_app.frontend.utilities.app_state_watcher import AppStateWatcher

__all__ = ["PanelFrontend", "AppStateWatcher"]
