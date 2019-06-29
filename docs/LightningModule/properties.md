A LightningModule has the following properties which you can access at any time

--- 
#### current_epoch
The current epoch   

---
#### dtype    
Current dtype    

---
#### experiment    
An instance of test-tube Experiment which you can use to log anything for tensorboarX.   
```{.python}
self.experiment.add_embedding(...)   
self.experiment.log({'val_loss': 0.9})   
self.experiment.add_scalars(...)   
```

--- 
#### global_step   
Total training batches seen across all epochs   

--- 
#### gradient_clip   
The current gradient clip value    

---
#### on_gpu    
True if your model is currently running on GPUs. Useful to set flags around the LightningModule for different CPU vs GPU behavior.    

---
#### trainer    
Last resort access to any state the trainer has. Changing certain properties here could affect your training run.
```{.python}   
self.trainer.optimizers
self.trainer.current_epoch
...   
```   

