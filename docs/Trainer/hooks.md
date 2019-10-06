# Hooks
[[Github Code](https://github.com/williamFalcon/pytorch-lightning/blob/master/pytorch_lightning/root_module/hooks.py)]   

There are cases when you might want to do something different at different parts of the training/validation loop.
To enable a hook, simply override the method in your LightningModule and the trainer will call it at the correct time.

**Contributing** If there's a hook you'd like to add, simply:    
1. Fork PyTorchLightning.    
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
#### on_epoch_end
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
#### on_batch_end
Called in the training loop after the batch.   
```python
def on_batch_end(self):
    # do something when the batch ends 
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
#### optimizer_step 
Calls .step() and .zero_grad for each optimizer.  
You can override this method to adjust how you do the optimizer step for each optimizer

Called once per optimizer
```python
# DEFAULT
def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
    optimizer.step()   
    optimizer.zero_grad()   
    
# Alternating schedule for optimizer steps (ie: GANs)    
def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
    # update generator opt every 2 steps
    if optimizer_i == 0:
        if batch_nb % 2 == 0 :
            optimizer.step()
            optimizer.zero_grad()
   
    # update discriminator opt every 4 steps
    if optimizer_i == 1:
        if batch_nb % 4 == 0 :
            optimizer.step()
            optimizer.zero_grad()    
    
    # ...
    # add as many optimizers as you want 
```

This step allows you to do a lot of non-standard training tricks such as learning-rate warm-up:   

```python
# learning rate warm-up
def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
    # warm up lr
    if self.trainer.global_step < 500:
        lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
        for pg in optimizer.param_groups:
            pg['lr'] = lr_scale * self.hparams.learning_rate
    
    # update params
    optimizer.step()
    optimizer.zero_grad() 
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
            self.logger.experiment.add_histogram(tag=name, values=grads, global_step=self.trainer.global_step)
```
