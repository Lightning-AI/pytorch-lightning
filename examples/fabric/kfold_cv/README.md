## K-Fold Cross Validation

This is an example of performing K-Fold cross validation supported with [Lightning Fabric](https://lightning.ai/docs/fabric). To learn more about cross validation, check out [this article](https://sebastianraschka.com/blog/2016/model-evaluation-selection-part3.html#introduction-to-k-fold-cross-validation).

We use the MNIST dataset to train a simple CNN model. We create the k-fold cross validation splits using the `ModelSelection.KFold` [class](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) in the `scikit-learn` library. Ensure that you have the `scikit-learn` library installed;

```bash
pip install scikit-learn
```

#### Run K-Fold Image Classification with Lightning Fabric

This script shows you how to scale the pure PyTorch code to enable GPU and multi-GPU training using [Lightning Fabric](https://lightning.ai/docs/fabric).

```bash
# CPU
fabric run train_fabric.py

# GPU (CUDA or M1 Mac)
fabric run train_fabric.py --accelerator=gpu

# Multiple GPUs
fabric run train_fabric.py --accelerator=gpu --devices=4
```

### References

- [KFold Model Selection](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
- [K-Fold Cross Validation by Sebastian Rashcka](https://sebastianraschka.com/blog/2016/model-evaluation-selection-part3.html#introduction-to-k-fold-cross-validation)
- [Cross Validation Wiki](<https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation>)
