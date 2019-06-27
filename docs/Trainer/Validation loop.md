The lightning validation loop handles everything except the actual computations of your model. To decide what will happen in your validation loop, define the [validation_step function](../../Pytorch-lightning/LightningModule/#validation_step).
Below are all the things lightning automates for you in the validation loop.

**Note**   
Lightning will run 5 steps of validation in the beginning of training as a sanity check so you don't have to wait until a full epoch to catch possible validation issues.




---
#### Check validation every n epochs
If you have a small dataset you might want to check validation every n epochs
``` {.python}
# DEFAULT
trainer = Trainer(check_val_every_n_epoch=1)
```

---
#### Set how much of the validation set to check 
If you don't want to check 100% of the validation set (for debugging or if it's huge), set this flag
``` {.python}
# DEFAULT
trainer = Trainer(val_percent_check=1.0)

# check 10% only
trainer = Trainer(val_percent_check=0.1)
```

---
#### Set how much of the test set to check 
If you don't want to check 100% of the test set (for debugging or if it's huge), set this flag
``` {.python}
# DEFAULT
trainer = Trainer(test_percent_check=1.0)

# check 10% only
trainer = Trainer(test_percent_check=0.1)
```

---
####  Set validation check frequency within 1 training epoch
For large datasets it's often desirable to check validation multiple times within a training loop
``` {.python}
# DEFAULT
trainer = Trainer(val_check_interval=0.95)

# check every .25 of an epoch 
trainer = Trainer(val_check_interval=0.25)
```

---
####  Set the number of validation sanity steps
Lightning runs a few steps of validation in the beginning of training. This avoids crashing in the validation loop sometime deep into a lengthy training loop.
``` {.python}
# DEFAULT
trainer = Trainer(nb_sanity_val_steps=5)
```