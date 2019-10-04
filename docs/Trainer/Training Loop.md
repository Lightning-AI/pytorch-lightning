The lightning training loop handles everything except the actual computations of your model. To decide what will happen in your training loop, define the [training_step function](https://williamfalcon.github.io/pytorch-lightning/LightningModule/RequiredTrainerInterface/#training_step).

Below are all the things lightning automates for you in the training loop.

---
#### Accumulated gradients  
Accumulated gradients runs K small batches of size N before doing a backwards pass. The effect is a large effective batch size of size KxN. 

``` {.python}
# DEFAULT (ie: no accumulated grads)
trainer = Trainer(accumulate_grad_batches=1)
```

---
#### Force training for min or max epochs
It can be useful to force training for a minimum number of epochs or limit to a max number
``` {.python}
# DEFAULT
trainer = Trainer(min_nb_epochs=1, max_nb_epochs=1000)
```

---
#### Early stopping
The trainer already sets up default early stopping for you. 
To modify this behavior, pass in your own EarlyStopping callback.
``` {.python}
from pytorch_lightning.callbacks import EarlyStopping

# DEFAULTS used by Trainer
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode='min'
)

trainer = Trainer(early_stop_callback=early_stop_callback)
```

---
#### Force disable early stop 
Use this to turn off early stopping and run training to the [max_epoch](#force-training-for-min-or-max-epochs)
``` {.python}
# DEFAULT
trainer = Trainer(enable_early_stop=True)
```

---
#### Gradient Clipping
Gradient clipping may be enabled to avoid exploding gradients.
Specifically, this will [clip the gradient norm computed over all model parameters *together*](https://pytorch.org/docs/stable/nn.html#torch.nn.utils.clip_grad_norm_).

``` {.python}
# DEFAULT (ie: don't clip)
trainer = Trainer(gradient_clip_val=0)

# clip gradients with norm above 0.5
trainer = Trainer(gradient_clip_val=0.5)
```

---
#### Inspect gradient norms
Looking at grad norms can help you figure out where training might be going wrong.
``` {.python}
# DEFAULT (-1 doesn't track norms)
trainer = Trainer(track_grad_norm=-1)

# track the LP norm (P=2 here)
trainer = Trainer(track_grad_norm=2)
```


---
#### Set how much of the training set to check
If you don't want to check 100% of the training set (for debugging or if it's huge), set this flag.

train_percent_check will be overwritten by overfit_pct if `overfit_pct > 0`

``` {.python}
# DEFAULT
trainer = Trainer(train_percent_check=1.0)

# check 10% only
trainer = Trainer(train_percent_check=0.1)
```
