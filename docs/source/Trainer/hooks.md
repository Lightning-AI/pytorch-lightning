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
#### backward
Called to perform backward step.
Feel free to override as needed.

The loss passed in has already been scaled for accumulated gradients if requested.
```python
def backward(self, use_amp, loss, optimizer):
    """
    Override backward with your own implementation if you need to
    :param use_amp: Whether amp was requested or not
    :param loss: Loss is already scaled by accumulated grads
    :param optimizer: Current optimizer being used
    :return:
    """
    if use_amp:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
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

---
#### tbptt_split_batch
Called in the training loop after on_batch_start if `truncated_bptt_steps > 0`. Each returned batch split is passed separately to training_step(...).

```python
def tbptt_split_batch(self, batch, split_size):
  splits = []
  for t in range(0, time_dims[0], split_size):
      batch_split = []
      for i, x in enumerate(batch):
          if isinstance(x, torch.Tensor):
              split_x = x[:, t:t + split_size]
          elif isinstance(x, collections.Sequence):
              split_x = [None] * len(x)
              for batch_idx in range(len(x)):
                  split_x[batch_idx] = x[batch_idx][t:t + split_size]

          batch_split.append(split_x)

      splits.append(batch_split)

  return splits
```

---
#### configure_apex
Overwrite to define your own Apex implementation init.

```python
def configure_apex(self, amp, model, optimizers, amp_level):
    """
    Override to init AMP your own way
    Must return a model and list of optimizers
    :param amp:
    :param model:
    :param optimizers:
    :param amp_level:
    :return: Apex wrapped model and optimizers
    """
    model, optimizers = amp.initialize(
        model, optimizers, opt_level=amp_level,
    )

    return model, optimizers
```

---
#### configure_ddp 
Overwrite to define your own DDP implementation init.
The only requirement is that:
1. On a validation batch the call goes to model.validation_step.   
2. On a training batch the call goes to model.training_step.   
3. On a testing batch, the call goes to model.test_step

```python
def configure_ddp(self, model, device_ids):
    """
    Override to init DDP in a different way or use your own wrapper.
    Must return model.
    :param model:
    :param device_ids:
    :return: DDP wrapped model
    """
    # Lightning DDP simply routes to test_step, val_step, etc...
    model = LightningDistributedDataParallel(
        model,
        device_ids=device_ids,
        find_unused_parameters=True
    )
    return model
```

---   
#### init_ddp_connection   
Override to init DDP in your own way.   

```python
def init_ddp_connection(self):
    """
    Connect all procs in the world using the env:// init
    Use the first node as the root address
    """

    # use slurm job id for the port number
    # guarantees unique ports across jobs from same grid search
    try:
        # use the last 4 numbers in the job id as the id
        default_port = os.environ['SLURM_JOB_ID']
        default_port = default_port[-4:]

        # all ports should be in the 10k+ range
        default_port = int(default_port) + 15000

    except Exception as e:
        default_port = 12910

    # if user gave a port number, use that one instead
    try:
        default_port = os.environ['MASTER_PORT']
    except Exception:
        os.environ['MASTER_PORT'] = str(default_port)

    # figure out the root node addr
    try:
        root_node = os.environ['SLURM_NODELIST'].split(' ')[0]
    except Exception:
        root_node = '127.0.0.2'

    root_node = self.trainer.resolve_root_node_address(root_node)
    os.environ['MASTER_ADDR'] = root_node
    dist.init_process_group('nccl', rank=self.proc_rank, world_size=self.world_size)
```
