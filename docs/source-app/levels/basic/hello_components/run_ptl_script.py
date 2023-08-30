# app.py
# !curl https://raw.githubusercontent.com/Lightning-AI/lightning/master/examples/app/multi_node/pl_boring_script.py -o pl_boring_script.py
from lightning.app import LightningApp, CloudCompute
from lightning.app.components.training import LightningTrainerScript

# run script that trains PyTorch with the Lightning Trainer
model_script = 'pl_boring_script.py'
component = LightningTrainerScript(
   model_script,
   num_nodes=1,
   cloud_compute=CloudCompute("gpu")
)
app = LightningApp(component)
