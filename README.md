# Pytorch-lightning
The Keras for ML-researchers in PyTorch.    

## Usage
To use lightning do 2 things:  
1. [Define a trainer](https://github.com/williamFalcon/pytorch-lightning/blob/master/pytorch_lightning/trainer_main.py) (which will run ALL your models).   
2. [Define a model](https://github.com/williamFalcon/pytorch-lightning/blob/master/pytorch_lightning/models/sample_model_template/model_template.py).     

### Example:   

#### Define the trainer   

```python
# trainer.py

from pytorch_lightning.models.trainer import Trainer   
from pytorch_lightning.utils.pt_callbacks import EarlyStopping, ModelCheckpoint
from my_project import My_Model   
from test_tube import HyperOptArgumentParser, Experiment, SlurmCluster

# --------------
# TEST TUBE INIT
exp = Experiment(
    name='my_exp',
    debug=True,
    save_dir='/some/path',
    autosave=False,
    description='my desc'
)

# --------------------
# CALLBACKS
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=True,
    mode='min'
)

model_save_path = 'PATH/TO/SAVE'
checkpoint = ModelCheckpoint(
    filepath=model_save_path,
    save_function=None,
    save_best_only=True,
    verbose=True,
    monitor='val_acc',
    mode='min'
)

# configure trainer
trainer = Trainer(
    experiment=experiment,
    cluster=cluster,
    checkpoint_callback=checkpoint,
    early_stop_callback=early_stop
)  

# init model and train
model = My_Model()
trainer.fit(model)
```

#### Define the model   

```python
import torch.nn as nn

class My_Model(RootModule):
    def __init__(self):
        # define model
    
    # ---------------
    # TRAINING
    def training_step(self, data_batch):
        x, y = data_batch
        y_hat = self.l1(x)
        loss = some_loss(y_hat)
        
        return loss_val, {'train_loss': loss}
    
    def validation_step(self, data_batch):
        x, y = data_batch
        y_hat = self.l1(x)
        loss = some_loss(y_hat)
        
        return loss_val, {'val_loss': loss}
 
     def validation_end(self, outputs):
        total_accs = []
        
        for output in outputs:
            total_accs.append(output['val_acc'].item())
        
        # return a dict
        return {'total_acc': np.mean(total_accs)}
     
     # ---------------
     # SAVING
     def get_save_dict(self):
        # lightning saves for you. Here's your chance to say what you want to save
        checkpoint = {'state_dict': self.state_dict()}

        return checkpoint

    def load_model_specific(self, checkpoint):
        # lightning loads for you. Here's your chance to say what you want to load
        self.load_state_dict(checkpoint['state_dict'])
    
    # ---------------
    # TRAINING CONFIG
    def configure_optimizers(self):
        # give lightning the list of optimizers you want to use.
        # lightning will call automatically
        optimizer = self.choose_optimizer('adam', self.parameters(), {'lr': self.hparams.learning_rate}, 'optimizer')
        return [optimizer]
    
    @property
    def tng_dataloader(self):
        return pytorch_dataloader('train')

    @property
    def val_dataloader(self):
        return pytorch_dataloader('val')

    @property
    def test_dataloader(self):
        return pytorch_dataloader('test')
        
    # ---------------
    # MODIFY YOUR COMMAND LINE ARGS
    @staticmethod
    def add_model_specific_args(parent_parser):    
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser])
        parser.add_argument('--out_features', default=20)
        return parser
```
### Details    

#### Model definition   
| Name  | Description  |  Input |  Return |
|---|---|---|---|
| training_step  |  Called with a batch of data during training | data from your dataloaders  | tuple: scalar, dict  |
| validation_step  |  Called with a batch of data during validation  | data from your dataloaders  | tuple: scalar, dict |
| validation_end  |  Collate metrics from all validation steps |  outputs: array where each item is the output of a validation step | dict: for logging |
| get_save_dict  | called when your model needs to be saved (checkpoints, hpc save, etc...)  | None  |  dict to be saved | 

#### Model training   
| Name  | Description  |  Input |  Return |
|---|---|---|---|
| configure_optimizers  |  called during training setup | None | list: optimizers you want to use |
| tng_dataloader  |  called during training  | None  | pytorch dataloader |
| val_dataloader  |  called during validation  | None  | pytorch dataloader |
| test_dataloader  |  called during testing  | None  | pytorch dataloader |    
| add_model_specific_args  |  called with args you defined in your main. This lets you tailor args for each model and keep main the same  | argparse  | argparse |

#### Model Saving/Loading   
| Name  | Description  |  Input |  Return |
|---|---|---|---|
| get_save_dict  | called when your model needs to be saved (checkpoints, hpc save, etc...)  | None  |  dict to be saved |
|  load_model_specific |  called when loading a model | checkpoint: dict you created in get_save_dict  | dict: modified in whatever way you want  |
    
## Optional model hooks.   
Add these to the model whenever you want to configure training behavior.    


### Model lifecycle hooks
Use these hooks to customize functionality

| Method | Purpose  | Input  | Output  | Required  |
|---|---|---|---|---|
| on_batch_start()  | called right before the batch starts | - | -  | N  |
| on_batch_end()  | called right after the batch ends | - | -  | N  |
| on_epoch_start()  | called right before the epoch starts | - | -  | N  |
| on_epoch_end()  | called right afger the epoch ends | - | -  | N  |
| on_pre_performance_check()  | called right before the performance check starts | - | -  | N  |
| on_post_performance_check()  | called right after the batch starts | - | -  | N  |
