To ensure you don't accidentally use test data to guide training decisions Lightning makes running the test set deliberate.   

---
#### test
You have two options to run the test set. 
First case is where you test right after a full training routine.
``` {.python}
# run full training
trainer.fit(model)

# run test set
trainer.test()
```   

Second case is where you load a model and run the test set   
```{.python}
model = MyLightningModule.load_from_metrics(
    weights_path='/path/to/pytorch_checkpoint.ckpt',
    tags_csv='/path/to/test_tube/experiment/version/meta_tags.csv',
    on_gpu=True,
    map_location=None
)

# init trainer with whatever options
trainer = Trainer(...)
    
# test (pass in the model)
trainer.test(model)
```
In this second case, the options you pass to trainer will be used when running the test set (ie: 16-bit, dp, ddp, etc...)  
 
