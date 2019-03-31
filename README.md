# Pytorch-lightning
Seed for ML research

## Usage
To use lightning, define a model that implements these 10 functions:

#### Model definition   
| Name  | Description  |  Input |  Return |
|---|---|---|---|
| training_step  |  Called with a batch of data during training | data from your dataloaders  | tuple: scalar, dict  |
| validation_step  |  Called with a batch of data during validation  | data from your dataloaders  | tuple: scalar, dict |
| validation_end  |  Collate metrics from all validation steps |  outputs: array where each item is the output of a validation step | dict: for logging |
| get_save_dict  | called when your model needs to be saved (checkpoints, hpc save, etc...)  | None  |  dict to be saved |
|  load_model_specific |   |   |   |    

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


## Example     
```python
import torch.nn as nn

class ExampleModel(RootModule):
    def __init__(self):
        self.l1 = nn.Linear(100, 20)
    
    # TRAINING
    def training_step(self, data_batch):
        # your dataloader decides what each batch looks like
        x, y = data_batch
        y_hat = self.l1(x)
        loss = some_loss(y_hat)
        
        tqdm_dic = {'train_loss': loss}
        
        # must return scalar, dict for logging
        return loss_val, tqdm_dic
    
    def validation_step(self, data_batch):
        # same as training...
        x, y = data_batch
        y_hat = self.l1(x)
        loss = some_loss(y_hat)
        
        # val specific
        acc = calculate_acc(y_hat, y)
        
        tqdm_dic = {'train_loss': loss, 'val_acc': acc, 'whatever_you_want': 'a'}
        return loss_val, tqdm_dic
 
     def validation_end(self, outputs):
        total_accs = []
        
        # given to you by the framework with all validation outputs.
        # chance to collate
        for output in outputs:
            total_accs.append(output['val_acc'].item())
        
        # return a dict
        return {'total_acc': np.mean(total_accs)}
        
     # SAVING
     def get_save_dict(self):
        # lightning saves for you. Here's your chance to say what you want to save
        checkpoint = {'state_dict': self.state_dict()}

        return checkpoint

    def load_model_specific(self, checkpoint):
        # lightning loads for you. Here's your chance to say what you want to load
        self.load_state_dict(checkpoint['state_dict'])
        pass
    
    # TRAINING CONFIG
    def configure_optimizers(self):
        # give lightning the list of optimizers you want to use.
        # lightning will call automatically
        optimizer = self.choose_optimizer(self.hparams.optimizer_name, self.parameters(), {'lr': self.hparams.learning_rate}, 'optimizer')
        self.optimizers = [optimizer]
        return self.optimizers
    
    # LIGHTING WILL USE THE LOADERS YOU DEFINE HERE
    @property
    def tng_dataloader(self):
        return pytorch_dataloader('train')

    @property
    def val_dataloader(self):
        return pytorch_dataloader('val')

    @property
    def test_dataloader(self):
        return pytorch_dataloader('test')
    
    # MODIFY YOUR COMMAND LINE ARGS
    @staticmethod
    def add_model_specific_args(parent_parser):    
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser])
        parser.add_argument('--out_features', default=20)
        return parser
```

### Add new model
1. Create a new model under /models.
2. Add model name to trainer_main
```python
AVAILABLE_MODELS = {
    'model_1': ExampleModel1
}
```

### Model methods that can be implemented

| Method | Purpose  | Input  | Output  | Required  |
|---|---|---|---|---|
| forward()  | Forward pass   | model_in tuple with your data  | model_out tuple to be passed to loss  | Y  |
| loss()  | calculate model loss  | model_out tuple from forward()  | A scalar  | Y  |
| check_performance()  | run a full loop through val data to check for metrics  | dataloader, nb_tests  | metrics tuple to be tracked  | Y  |
| tng_dataloader  | Computed option, used to feed tng data  | -  | Pytorch DataLoader subclass  | Y  |
| val_dataloader  | Computed option, used to feed tng data  | -  | Pytorch DataLoader subclass  | Y  |
| test_dataloader  | Computed option, used to feed tng data  | -  | Pytorch DataLoader subclass  | Y  |

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
