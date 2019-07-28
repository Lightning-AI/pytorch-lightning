# Hooks
[[Github Code](https://github.com/williamFalcon/pytorch-lightning/blob/master/pytorch_lightning/root_module/hooks.py)]   

There are cases when you might want to do something different at different parts of the training/validation loop.
To enable a hook, simply override the method in your LightningModule and the trainer will call it at the correct time.

**Contributing** If there's a hook you'd like to add, simply:    
1. Fork PytorchLightning.    
2. Add the hook [here](https://github.com/williamFalcon/pytorch-lightning/blob/master/pytorch_lightning/root_module/hooks.py).       
3. Add the correct place in the [Trainer](https://github.com/williamFalcon/pytorch-lightning/blob/master/pytorch_lightning/models/trainer.py) where it should be called.    

---
#### on_epoch_start
Called in the training loop at the very beginning of the epoch.   
```python
def on_epoch_start(self):
    # do something when the epoch starts
```

---
#### on_batch_end
Called in the training loop at the very end of the epoch.   
```python
def on_epoch_end(self):
    # do something when the epoch ends 
```

---
#### on_batch_start
Called in the training loop before anything happens for that batch.   
```python
def on_batch_start(self):
    # do something when the batch starts
```

---
#### on_pre_performance_check
Called at the very beginning of the validation loop.   
```python
def on_pre_performance_check(self):
    # do something before validation starts 
```

---
#### on_post_performance_check
Called at the very end of the validation loop.   
```python
def on_post_performance_check(self):
    # do something before validation end
```

---
#### on_tng_metrics
Called in the training loop, right before metrics are logged.
Although you can log at any time by using self.experiment, you can use
this callback to modify what will be logged.
```python
def on_tng_metrics(self, metrics):
    # do something before validation end
```

---
#### on_before_zero_grad
Called in the training loop after taking an optimizer step and before zeroing grads.
Good place to inspect weight information with weights updated.

Called once per optimizer
```python
def on_before_zero_grad(self, optimizer):
    # do something with the optimizer or inspect it. 
```

---
#### on_after_backward
Called in the training loop after model.backward()
This is the ideal place to inspect or log gradient information 
```python
def on_after_backward(self):
    # example to inspect gradient information in tensorboard
    if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
        params = self.state_dict()
        for k, v in params.items():
            grads = v
            name = k
            self.experiment.add_histogram(tag=name, values=grads, global_step=self.trainer.global_step)
```
