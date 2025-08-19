## Transformers

This example contains a simple training loop for next-word prediction with a [Transformer model](https://arxiv.org/abs/1706.03762) on a subset of the [WikiText2](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) dataset.
It is a simplified version of the [official PyTorch example](https://github.com/pytorch/examples/tree/main/word_language_model).

### Train with Fabric

```bash
# CPU
fabric run --accelerator=cpu train.py

# GPU (CUDA or M1 Mac)
fabric run --accelerator=gpu train.py

# Multiple GPUs
fabric run --accelerator=gpu --devices=4 train.py
```
