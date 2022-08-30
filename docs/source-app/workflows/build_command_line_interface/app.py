from commands.notebook.run import RunNotebook, RunNotebookConfig
from lit_jupyter import JupyterLab

import lightning as L
from lightning.app.structures import Dict


class Flow(L.LightningFlow):

    def __init__(self):
        super().__init__()
        self.notebooks = Dict()

    # 1. Annotate the handler input with the notebook config.
    def run_notebook(self, config: RunNotebookConfig):
        if config.name in self.notebooks:
            return f"The notebook {config.name} is already created."
        else:
            # 2. Create dynamically the notebook if it doesn't exist and run it.
            self.notebooks[config.name] = JupyterLab(
                cloud_compute=L.CloudCompute(config.cloud_compute)
            )
            self.notebooks[config.name].run()
            return f"The notebook {config.name} was created."

    def configure_commands(self):
        # 3. Returns a list of dictionaries with the format:
        # {"command_name": CustomClientCommand(method=self.custom_server_handler)}
        return [{"run-notebook": RunNotebook(method=self.run_notebook)}]

    def configure_layout(self):
        # 4. Dynamically display the notebooks in the Lightning App View.
        return [{"name": n, "content": w} for n, w in self.notebooks.items()]


app = L.LightningApp(Flow())
