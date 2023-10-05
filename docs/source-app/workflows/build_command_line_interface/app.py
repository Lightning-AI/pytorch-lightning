from commands.notebook.run import RunNotebook, RunNotebookConfig
from lit_jupyter import JupyterLab

from lightning.app import LightningFlow, LightningApp, CloudCompute
from lightning.app.structures import Dict


class Flow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.notebooks = Dict()

    # 1. Annotates the handler input with the Notebook config.
    def run_notebook(self, config: RunNotebookConfig):
        if config.name in self.notebooks:
            return f"The Notebook {config.name} already exists."
        else:
            # 2. Dynamically creates the Notebook if it doesn't exist and runs it.
            self.notebooks[config.name] = JupyterLab(
                cloud_compute=CloudCompute(config.cloud_compute)
            )
            self.notebooks[config.name].run()
            return f"The Notebook {config.name} was created."

    def configure_commands(self):
        # 3. Returns a list of dictionaries with the format:
        # {"command_name": CustomClientCommand(method=self.custom_server_handler)}
        return [{"run notebook": RunNotebook(method=self.run_notebook)}]

    def configure_layout(self):
        # 4. Dynamically displays the Notebooks in the Lightning App View.
        return [{"name": n, "content": w} for n, w in self.notebooks.items()]


app = LightningApp(Flow())
