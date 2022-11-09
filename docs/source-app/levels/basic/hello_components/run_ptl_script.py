# EXAMPLE COMPONENT: RUN A SCRIPT
# app.py
import lightning as L
from lightning.app.components.training import LightningTrainingComponent


# run script that trains PyTorch with the Lightning Trainer
model_script = 'lightning_trainer_script.py'
component = LightningTrainingComponent(
   model_script, 
   num_nodes=1,
   cloud_compute=L.CloudCompute("gpu")
)
app = L.LightningApp(component)
