# Pytorch-lightning
Seed for ML research

## Usage

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
