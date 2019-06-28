A LightningModule has the following properties which you can access at any time

--- 
#### current_epoch
The current epoch   

---
#### dtype    
Current dtype    

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
#### Trainer
Last resort access to any state the trainer has. Changing certain properties here could affect your training run.
