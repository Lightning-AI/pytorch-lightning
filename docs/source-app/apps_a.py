# app.py
from mingpt.trainer import Trainer
from mingpt.model import GPT
import lightning as L


class GPTTrainingComponent(L.LightningWork):
   """
   Component to train a GPT-3 model (https://github.com/karpathy/minGPT)
   But you can replace with your own training code written in ANY ML framework  
   """
   def __init__(self) -> None:
      super().__init__()
      self.cloud_build_config = L.BuildConfig(requirements=["https://github.com/karpathy/minGPT.git"])

   def run(self):
      model_config = GPT.get_default_config()
      model_config.model_type = 'gpt2'
      model_config.vocab_size = 50257 
      model_config.block_size = 1024  
      
      model = GPT(model_config)
      train_dataset = YourDataset() # TODO: (synthetic data?)

      train_config = Trainer.get_default_config()
      train_config.learning_rate = 5e-4
      train_config.max_iters = 1000
      train_config.batch_size = 32
      trainer = Trainer(train_config, model, train_dataset)
      trainer.run()

# run on a cloud machine with 8 GPUs
compute = L.CloudCompute("gpu-multi-fast")
worker = GPTComponent(cloud_compute=compute)
app = L.LightningApp(worker)