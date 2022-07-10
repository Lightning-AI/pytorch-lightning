import os
from unittest.mock import Mock

import panel as pn
import param
from app_state_watcher import AppStateWatcher

from lightning_app.utilities.state import AppState

def to_str(value):
    if isinstance(value, list):
        return "\n".join([str(item) for item in value])
    return str(value)

if "LIGHTNING_FLOW_NAME" in os.environ:
    app = AppStateWatcher()
else:
    class AppMock(param.Parameterized):
        state = param.Parameter()

    app = AppMock(state=Mock())
    import json

    with open("state.json", "r") as fp:
        app.state._state = json.load(fp)
    app.state.requests = [
        {
            "id": 0,
            "train": {
                "github_repo": "https://github.com/Lightning-AI/lightning-quick-start.git",
                "script_path": "train_script.py",
                "script_args": [
                    "--trainer.max_epochs=5",
                    "--trainer.limit_train_batches=4",
                    "--trainer.limit_val_batches=4",
                    "--trainer.callbacks=ModelCheckpoint",
                    "--trainer.callbacks.monitor=val_acc",
                ],
                "requirements": ["torchvision,", "pytorch_lightning,", "jsonargparse[signatures]"],
                "ml_framework": "PyTorch Lightning",
            },
        }
    ]
    

ACCENT = "#792EE5"
LIGHTNING_SPINNER_URL = (
    "https://cdn.jsdelivr.net/gh/MarcSkovMadsen/awesome-panel-assets@master/spinners/material/"
    "bar_chart_lightning_purple.svg"
)
LIGHTNING_SPINNER = pn.pane.HTML(
    f"<img src='{LIGHTNING_SPINNER_URL}' style='height:100px;width:100px;'/>"
)
# Todo: Set JSON theme depending on template theme
# pn.pane.JSON.param.theme.default = "dark"
pn.pane.JSON.param.hover_preview.default = True

pn.config.raw_css.append(
    """
  .bk-root {
    height: calc( 100vh - 200px ) !important;
  }
  """
)
pn.extension("terminal", sizing_mode="stretch_width", template="fast")
pn.state.template.param.update(accent_base_color=ACCENT, header_background=ACCENT)


def create_new_page():
    title = "# Create a new run 🎈"
    id_input = pn.widgets.TextInput(name="Name your run", value="my_first_run")
    github_repo_input = pn.widgets.TextInput(
        name="Enter a Github Repo URL",
        value="https://github.com/Lightning-AI/lightning-quick-start.git",
    )
    script_path_input = pn.widgets.TextInput(
        name="Enter your script to run", value="train_script.py"
    )

    default_script_args = "--trainer.max_epochs=5 --trainer.limit_train_batches=4 --trainer.limit_val_batches=4 --trainer.callbacks=ModelCheckpoint --trainer.callbacks.monitor=val_acc"
    script_args_input = pn.widgets.TextInput(
        name="Enter your base script arguments", value=default_script_args
    )
    default_requirements = "torchvision, pytorch_lightning, jsonargparse[signatures]"
    requirements_input = pn.widgets.TextInput(
        name="Enter your requirements", value=default_requirements
    )
    ml_framework_input = pn.widgets.RadioBoxGroup(
        name="Select your ML Training Frameworks",
        options=["PyTorch Lightning", "Keras", "Tensorflow"],
        inline=True,
    )
    submit_input = pn.widgets.Button(name="⚡ SUBMIT ⚡", button_type="primary")

    @pn.depends(submit_input, watch=True)
    def create_new_run(_):
        new_request = {
            "id": id_input.value,
            "train": {
                "github_repo": github_repo_input.value,
                "script_path": script_path_input.value,
                "script_args": script_args_input.value.split(" "),
                "requirements": requirements_input.value.split(" "),
                "ml_framework": ml_framework_input.value,
            },
        }
        app.state.requests = app.state.requests + [new_request]
        print("submitted", new_request)

    @pn.depends(ml_framework_input.param.value)
    def message_or_button(ml_framework):
        if ml_framework not in ("PyTorch Lightning"):
            return f"{ml_framework} isn't supported yet."
        else:
            return submit_input

    return pn.Column(
        title,
        id_input,
        github_repo_input,
        script_path_input,
        script_args_input,
        requirements_input,
        ml_framework_input,
        message_or_button,
    )


def card_show_work(idx, request, state):
    work = state["structures"]["ws"]["works"][f"w_{idx}"]

    def get_work_state():
        w = work["vars"].copy()
        if "logs" in w:
            w.pop("logs")
        return pn.pane.JSON(w, theme="light", sizing_mode="stretch_both")

    options = {
        "Expand to view your configuration": pn.pane.JSON(
            request, theme="light", hover_preview=True, depth=4
        ),
        "Expand to view logs": pn.Column(
            pn.pane.Markdown(
                "```bash\n" + to_str(work["vars"]["logs"]) + "\n```",
            ),
            height=800,
        ),
        "Expand to view your work state": get_work_state(),
    }
    selection_input = pn.widgets.RadioBoxGroup(name="Hello", options=list(options.keys()))

    @pn.depends(selection_input)
    def selection_output(value):
        return pn.panel(options[value], sizing_mode="stretch_both")

    return pn.Column(
        selection_input, selection_output, sizing_mode="stretch_both", name=f"Run: {idx}"
    )


@pn.depends(app.param.state)
def view_run_lists_page(state: AppState):
    title = "# Run Lists 🎈"
    # Todo: Consider other layout than accordion. Don't think its that great
    layout = pn.Accordion(sizing_mode="stretch_both")
    print(state._request_state)
    for idx, request in enumerate(state.requests):
        layout.append(card_show_work(idx, request, state._state))
        layout.append(card_show_work(idx, request, state._state))
    return pn.Column(title, layout)


@pn.depends(app.param.state)
def app_state_page(state: AppState):
    title = "# App State 🎈"
    # Todo: Make this on stretch full heigh of its parent containe
    json_output = pn.pane.JSON(state._state, theme="light", depth=6, max_height=800)
    return pn.Column(title, json_output, scroll=True)


pn.Tabs(
    ("New Run", create_new_page),
    ("View your Runs", view_run_lists_page),
    ("App State", app_state_page),
    sizing_mode="stretch_both",
).servable()
