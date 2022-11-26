import inspect
import warnings
from typing import Dict, List, Union

import lightning_app
from lightning_app.frontend.frontend import Frontend
from lightning_app.utilities.app_helpers import _MagicMockJsonSerializable
from lightning_app.utilities.cloud import is_running_in_cloud


def _add_comment_to_literal_code(method, contains, comment):
    """Inspects a method's code and adds a message to it.

    This is a nice to have, so if it fails for some reason, it shouldn't affect the program.
    """
    try:
        lines = inspect.getsource(method)
        lines = lines.split("\n")
        idx_list = [i for i, x in enumerate(lines) if contains in x]
        for i in idx_list:
            line = lines[i]
            line += comment
            lines[i] = line

        return "\n".join(lines)

    except Exception as e:  # noqa
        return ""


def _collect_layout(app: "lightning_app.LightningApp", flow: "lightning_app.LightningFlow") -> Union[Dict, List[Dict]]:
    """Process the layout returned by the ``configure_layout()`` method in each flow."""
    layout = flow.configure_layout()

    if isinstance(layout, Frontend):
        frontend = layout
        frontend.flow = flow
        app.frontends.setdefault(flow.name, frontend)

        # When running locally, the target will get overwritten by the dispatcher when launching the frontend servers
        # When running in the cloud, the frontend code will construct the URL based on the flow name
        return flow._layout
    elif isinstance(layout, _MagicMockJsonSerializable):
        # Do nothing
        pass
    elif isinstance(layout, dict):
        layout = _collect_content_layout([layout], flow)
    elif isinstance(layout, (list, tuple)) and all(isinstance(item, dict) for item in layout):
        layout = _collect_content_layout(layout, flow)
    else:
        lines = _add_comment_to_literal_code(flow.configure_layout, contains="return", comment="  <------- this guy")
        m = f"""
        The return value of configure_layout() in `{flow.__class__.__name__}`  is an unsupported layout format:
        \n{lines}

        Return either an object of type {Frontend} (e.g., StreamlitFrontend, StaticWebFrontend):
            def configure_layout(self):
                return la.frontend.Frontend(...)

        OR a single dict:
            def configure_layout(self):
                tab1 = {{'name': 'tab name', 'content': self.a_component}}
                return tab1

        OR a list of dicts:
            def configure_layout(self):
                tab1 = {{'name': 'tab name 1', 'content': self.component_a}}
                tab2 = {{'name': 'tab name 2', 'content': self.component_b}}
                return [tab1, tab2]

        (see the docs for `LightningFlow.configure_layout`).
        """
        raise TypeError(m)

    return layout


def _collect_content_layout(layout: List[Dict], flow: "lightning_app.LightningFlow") -> List[Dict]:
    """Process the layout returned by the ``configure_layout()`` method if the returned format represents an
    aggregation of child layouts."""
    for entry in layout:
        if "content" not in entry:
            raise ValueError(
                f"A dictionary returned by `{flow.__class__.__name__}.configure_layout()` is missing a key 'content'."
                f" For the value, choose either a reference to a child flow or a URla."
            )
        if isinstance(entry["content"], str):  # assume this is a URL
            url = entry["content"]
            if url.startswith("/"):
                # The URL isn't fully defined yet. Looks something like ``self.work.url + /something``.
                entry["target"] = ""
            else:
                entry["target"] = url
            if url.startswith("http://") and is_running_in_cloud():
                warnings.warn(
                    f"You configured an http link {url[:32]}... but it won't be accessible in the cloud."
                    f" Consider replacing 'http' with 'https' in the link above."
                )

        elif isinstance(entry["content"], lightning_app.LightningFlow):
            entry["content"] = entry["content"].name

        elif isinstance(entry["content"], lightning_app.LightningWork):
            if entry["content"].url and not entry["content"].url.startswith("/"):
                entry["content"] = entry["content"].url
                entry["target"] = entry["content"]
            else:
                entry["content"] = ""
                entry["target"] = ""
        elif isinstance(entry["content"], _MagicMockJsonSerializable):
            # Do nothing
            pass
        else:
            m = f"""
            A dictionary returned by `{flow.__class__.__name__}.configure_layout()` contains an unsupported entry.

            {{'content': {repr(entry['content'])}}}

            Set the `content` key to a child flow or a URL, for example:

            class {flow.__class__.__name__}(LightningFlow):
                def configure_layout(self):
                    return {{'content': childFlow OR childWork OR 'http://some/url'}}
            """
            raise ValueError(m)
    return layout
