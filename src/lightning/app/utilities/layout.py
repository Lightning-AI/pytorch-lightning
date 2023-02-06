# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import warnings
from typing import Dict, List, Union

import lightning.app
from lightning.app.frontend.frontend import Frontend
from lightning.app.utilities.app_helpers import _MagicMockJsonSerializable, is_overridden
from lightning.app.utilities.cloud import is_running_in_cloud


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


def _collect_layout(app: "lightning.app.LightningApp", flow: "lightning.app.LightningFlow") -> Union[Dict, List[Dict]]:
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
        # The import was mocked, we set a dummy `Frontend` so that `is_headless` knows there is a UI
        app.frontends.setdefault(flow.name, "mock")
        return flow._layout
    elif isinstance(layout, dict):
        layout = _collect_content_layout([layout], app, flow)
    elif isinstance(layout, (list, tuple)) and all(isinstance(item, dict) for item in layout):
        layout = _collect_content_layout(layout, app, flow)
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


def _collect_content_layout(
    layout: List[Dict], app: "lightning.app.LightningApp", flow: "lightning.app.LightningFlow"
) -> Union[List[Dict], Dict]:
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

        elif isinstance(entry["content"], lightning.app.LightningFlow):
            entry["content"] = entry["content"].name

        elif isinstance(entry["content"], lightning.app.LightningWork):
            work = entry["content"]
            work_layout = _collect_work_layout(work)

            if work_layout is None:
                entry["content"] = ""
            elif isinstance(work_layout, str):
                entry["content"] = work_layout
                entry["target"] = work_layout
            elif isinstance(work_layout, (Frontend, _MagicMockJsonSerializable)):
                if len(layout) > 1:
                    lines = _add_comment_to_literal_code(
                        flow.configure_layout, contains="return", comment="  <------- this guy"
                    )
                    m = f"""
                    The return value of configure_layout() in `{flow.__class__.__name__}`  is an
                    unsupported format:
                    \n{lines}

                    The tab containing a `{work.__class__.__name__}` must be the only tab in the
                    layout of this flow.

                    (see the docs for `LightningWork.configure_layout`).
                    """
                    raise TypeError(m)

                if isinstance(work_layout, Frontend):
                    # If the work returned a frontend, treat it as belonging to the flow.
                    # NOTE: This could evolve in the future to run the Frontend directly in the work machine.
                    frontend = work_layout
                    frontend.flow = flow
                elif isinstance(work_layout, _MagicMockJsonSerializable):
                    # The import was mocked, we set a dummy `Frontend` so that `is_headless` knows there is a UI.
                    frontend = "mock"

                app.frontends.setdefault(flow.name, frontend)
                return flow._layout

        elif isinstance(entry["content"], _MagicMockJsonSerializable):
            # The import was mocked, we just record dummy content so that `is_headless` knows there is a UI
            entry["content"] = "mock"
            entry["target"] = "mock"
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


def _collect_work_layout(work: "lightning.app.LightningWork") -> Union[None, str, Frontend, _MagicMockJsonSerializable]:
    """Check if ``configure_layout`` is overridden on the given work and return the work layout (either a string, a
    ``Frontend`` object, or an instance of a mocked import).

    Args:
        work: The work to collect the layout for.

    Raises:
        TypeError: If the value returned by ``configure_layout`` is not of a supported format.
    """
    if is_overridden("configure_layout", work):
        work_layout = work.configure_layout()
    else:
        work_layout = work.url

    if work_layout is None:
        return None
    elif isinstance(work_layout, str):
        url = work_layout
        # The URL isn't fully defined yet. Looks something like ``self.work.url + /something``.
        if url and not url.startswith("/"):
            return url
        return ""
    elif isinstance(work_layout, (Frontend, _MagicMockJsonSerializable)):
        return work_layout
    else:
        m = f"""
        The value returned by `{work.__class__.__name__}.configure_layout()` is of an unsupported type.

        {repr(work_layout)}

        Return a `Frontend` or a URL string, for example:

        class {work.__class__.__name__}(LightningWork):
            def configure_layout(self):
                return MyFrontend() OR 'http://some/url'
        """
        raise TypeError(m)
