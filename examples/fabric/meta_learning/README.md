## Meta-Learning - MAML

This is an example of a meta-learning algorithm called [MAML](https://arxiv.org/abs/1703.03400), trained on the
[Omniglot dataset](https://paperswithcode.com/dataset/omniglot-1) of handwritten characters from different alphabets.

The goal of meta-learning in this context is to learn a 'meta'-model trained on many different tasks, such that it can quickly adapt to a new task when trained with very few samples (few-shot learning).
If you are new to meta-learning, have a look at this short [introduction video](https://www.youtube.com/watch?v=ItPEBdD6VMk).

We show two code versions:
The first one is implemented in raw PyTorch, but it contains quite a bit of boilerplate code for distributed training.
The second one is using [Lightning Fabric](https://lightning.ai/docs/fabric) to accelerate and scale the model.

Tip: You can easily inspect the difference between the two files with:

```bash
sdiff train_torch.py train_fabric.py
```

### Requirements

```bash
pip install lightning learn2learn cherry-rl 'gym<=0.22'
```

### Run

**Raw PyTorch:**

```bash
torchrun --nproc_per_node=2 --standalone train_torch.py
```

**Accelerated using Lightning Fabric:**

```bash
fabric run train_fabric.py --devices 2 --strategy ddp --accelerator cpu
```

### References

- [MAML explained in 7 minutes](https://www.youtube.com/watch?v=ItPEBdD6VMk)
- [Learn2Learn Resources](http://learn2learn.net/examples/vision/#maml)
- [MAML Paper](https://arxiv.org/abs/1703.03400)
