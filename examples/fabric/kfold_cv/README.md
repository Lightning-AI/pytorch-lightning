## K-Fold Cross Validation

This is an example of performing K-Fold cross validation supported with [Lightning Fabric](https://pytorch-lightning.readthedocs.io/en/latest/fabric/fabric.html). To know more about cross validation, check out [this article](https://sebastianraschka.com/blog/2016/model-evaluation-selection-part3.html#introduction-to-k-fold-cross-validation).

We use the MNIST dataset to train a simple CNN model. We create the k-fold cross validation splits using the `ModelSelection.KFold` [class](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) in the `scikit-learn` library. Ensure that you have the `scikit-learn` library installed.

### Run

**Accelerated using Lightning Fabric:**

```bash
python train_fabric.py
```

### References

- [KFold Model Selection](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
- [K-Fold Cross Validation by Sebastian Rashcka](https://sebastianraschka.com/blog/2016/model-evaluation-selection-part3.html#introduction-to-k-fold-cross-validation)
- [Cross Validation Wiki](<https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation>)
