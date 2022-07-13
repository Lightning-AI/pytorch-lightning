import json
import os
from functools import partial
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

pn.config.raw_css.append(
    """
  .bk-root {
    height: calc( 100vh - 200px ) !important;
  }
  .state-container {
      height: calc(100vh - 300px) !important;
  }
  .log-container {
      height: calc(100vh - 380px) !important;
  }
  .scrollable {
      overflow-x: hidden !important;
      overflow-y: scroll !important;
  }
  """
)

pn.extension("terminal", sizing_mode="stretch_width", template="fast", notifications=True)
pn.state.template.param.update(site="Panel Lightning ⚡", title="Github Model Runner", accent_base_color=ACCENT, header_background=ACCENT)
pn.pane.JSON.param.hover_preview.default = True

#region: Panel extensions

def _to_value(value):
    if hasattr(value, "value"):
        return value.value
    return value

def bind_as_form(function, *args, submit, watch=False, **kwargs):
    """Extends pn.bind to support "Forms" like binding. I.e. triggering only when a Submit button is clicked,
    but using the dynamic values of widgets or Parameters as inputs.
    
    Args:
        function (_type_): The function to execute
        submit (_type_): The Submit widget or parameter to depend on
        watch (bool, optional): Defaults to False.

    Returns:
        _type_: A Reactive Function
    """
    if not args:
        args = []
    if not kwargs:
        kwargs = {}

    def function_wrapper(_, args=args, kwargs=kwargs):
        args=[_to_value[value] for value in args]
        kwargs={key: _to_value(value) for key, value in kwargs.items()}
        return function(*args, **kwargs)
    return pn.bind(function_wrapper, submit, watch=watch)

def show_value(widget):
    """Shows the value of the widget or Parameter in a Panel
    
    Dynamically updated when ever the value changes
    """
    def show(value):
        return pn.panel(value, sizing_mode="stretch_both")    
    
    return pn.bind(show, value=widget)

THEME = pn.state.session_args.get("theme", [b"default"])[0].decode()
pn.pane.JSON.param.theme.default = THEME if THEME=="dark" else "light"


#endregion: Panel extensions
#region: Create new run

def create_new_run(id, github_repo, script_path, script_args, requirements, ml_framework):
    new_request = {
        "id": id,
        "train": {
            "github_repo": github_repo,
            "script_path": script_path,
            "script_args": script_args.split(" "),
            "requirements": requirements.split(" "),
            "ml_framework": ml_framework,
        },
    }
    app.state.requests = app.state.requests + [new_request]
    pn.state.notifications.send("New run created", background=ACCENT, icon='⚡')

def message_or_button(ml_framework, submit_button):
    if ml_framework not in ("PyTorch Lightning"):
        return f"💥 {ml_framework} isn't supported yet."
    else:
        return submit_button


def create_new_page():
    id_input = pn.widgets.TextInput(name="Name your run", value="my_first_run")
    github_repo_input = pn.widgets.TextInput(
        name="Enter a Github Repo URL",
        value="https://github.com/Lightning-AI/lightning-quick-start.git",
    )
    script_path_input = pn.widgets.TextInput(name="Enter your script to run", value="train_script.py")

    script_args_input = pn.widgets.TextInput(
        name="Enter your base script arguments",
        value=(
            "--trainer.max_epochs=5 --trainer.limit_train_batches=4 --trainer.limit_val_batches=4 "
            "--trainer.callbacks=ModelCheckpoint --trainer.callbacks.monitor=val_acc"
        ),
    )
    requirements_input = pn.widgets.TextInput(
        name="Enter your requirements", value="torchvision, pytorch_lightning, jsonargparse[signatures]"
    )
    ml_framework_input = pn.widgets.RadioBoxGroup(
        name="Select your ML Training Frameworks",
        options=["PyTorch Lightning", "Keras", "Tensorflow"],
        inline=True,
    )
    submit_button = pn.widgets.Button(name="⚡ SUBMIT ⚡", button_type="primary")
    bind_as_form(
        create_new_run,
        id=id_input,
        github_repo=github_repo_input,
        script_path=script_path_input,
        script_args=script_args_input,
        requirements=requirements_input,
        ml_framework=ml_framework_input,
        submit=submit_button,
        watch=True
    )

    return pn.Column(
        "# Create a new run 🎈",
        id_input,
        github_repo_input,
        script_path_input,
        script_args_input,
        requirements_input,
        ml_framework_input,
        pn.bind(partial(message_or_button, submit_button=submit_button), ml_framework_input),    
    )

#endregion: Create new run
#region: Run list page

def configuration_component(request):
    return pn.pane.JSON(request, depth=4)

def work_state_ex_logs_component(work):
    w = work["vars"].copy()
    if "logs" in w:
        w.pop("logs")
    return pn.pane.JSON(w, depth=4)

def log_component(work):
    return pn.Column(
        pn.pane.Markdown(
            "```bash\n" + to_str(work["vars"]["logs"]) + "\n```", max_height=500
        ), scroll=True, css_classes=["log-container"]     
    )

def run_component(idx, request, state):
    work = state["structures"]["ws"]["works"][f"w_{idx}"]
    name=work["vars"]["id"]
    return pn.Tabs(
        ("Configuration", configuration_component(request)),
        ("Work state", work_state_ex_logs_component(work)),
        ("Logs", log_component(work)),
        name=f"Run {idx}: {name}", margin=(5,0,0,0)
    )


@pn.depends(app.param.state)
def view_run_list_page(state: AppState):
    title = "# View your runs 🎈"
    layout = pn.Tabs(sizing_mode="stretch_both")
    for idx, request in enumerate(state.requests):
        layout.append(run_component(idx, request, state._state))       
    return pn.Column(title, layout)

#endregion: Run list page
#region: App state page


@pn.depends(app.param.state)
def view_app_state_page(state: AppState):
    title = "# View the full state of the app 🎈"
    json_output = pn.pane.JSON(state._state, depth=6)
    return pn.Column(title, pn.Column(json_output, scroll=True, css_classes=["state-container"]))

#endregion: App state page
#region: App
pn.Tabs(
    ("New Run", create_new_page),
    ("View Runs", view_run_list_page),
    ("View State", view_app_state_page),
    sizing_mode="stretch_both",
).servable()
#endregion: App
