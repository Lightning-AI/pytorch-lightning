:orphan:

.. _post_training_quantization:

##########################
Post-training Quantization
##########################

Most deep learning applications are using 32-bits of floating-point precision for inference. But low precision data types, especially INT8, are attracting more attention due to significant performance margin. One of the essential concerns of adopting low precision is how to easily mitigate the possible accuracy loss and reach predefined accuracy requirements.

Intel® Neural Compressor, is an open-source Python library that runs on Intel CPUs and GPUs, which could address the aforementioned concern by extending the PyTorch Lightning model with accuracy-driven automatic quantization tuning strategies to help users quickly find out the best-quantized model on Intel hardware. It also supports multiple popular network compression technologies such as sparse, pruning, and knowledge distillation.

**Audience** : Machine learning engineers optimizing models for a better model inference speed and lower memory usage.

Visit the Intel® Neural Compressor online document website at: `<https://intel.github.io/neural-compressor>`_.

******************
Model Quantization
******************

Model quantization is an efficient model optimization tool that can accelerate the model inference speed and decrease the memory load while still maintaining the model accuracy.

Intel® Neural Compressor provides a convenient model quantization API to quantize the already-trained Lightning module with Post-training Quantization and Quantization Aware Training. This extension API exhibits the merits of an ease-of-use coding environment and multi-functional quantization options. The user can easily quantize their fine-tuned model by adding a few clauses to their original code.  We only introduce post-training quantization in this document.

There are two post-training quantization types in Intel® Neural Compressor, post-training static quantization and post-training dynamic quantization.  Post-training dynamic quantization is a recommended starting point because it provides reduced memory usage and faster computation without additional calibration datasets. This type of quantization statically quantizes only the weights from floating point to integer at conversion time. This optimization provides latencies close to post-training static quantization. But the outputs of ops are still stored with the floating point, so the increased speed of dynamic-quantized ops is less than a static-quantized computation.

Post-training static quantization saves the output of ops via INT8 bit. It can tackle the accuracy and latency loss caused by "quant" and "dequant" operations. For Post-training static quantization, the user needs to estimate the min-max range of all FP32 tensors in the model. Unlike constant tensors such as weights and biases, variable tensors such as model input, activations and model output cannot be calibrated unless the model run a few inference cycles. As a result, the converter requires a calibration dataset to estimate that range. This dataset can be a small subset (default 100 samples) of the training or the validation data.

************
Installation
************

Prerequisites
=============

Python version: 3.8, 3.9, 3.10

Install Intel® Neural Compressor
================================

Release binary install:

.. code-block:: bash

    # Install stable basic version from pip
    pip install neural-compressor
    # Or install stable full version from pip (including GUI)
    pip install neural-compressor-full

More installation methods can be found in the `Installation Guide <https://github.com/intel/neural-compressor/blob/master/docs/source/installation_guide.md>`_.

*****
Usage
*****

Minor code changes are required for the user to get started with Intel® Neural Compressor quantization API. To construct the quantization process, users can specify the below settings via the Python code:

1. Calibration Dataloader (Needed for post-training static quantization)
2. Evaluation Dataloader and Metric

The code changes that are required for Intel® Neural Compressor are highlighted with comments in the line above.

PyTorch Lightning model
=======================

Load the pretrained model with PyTorch Lightning:

.. code-block:: python

    import torch
    from lightning.pytorch import LightningModule
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


    # BERT Model definition
    class GLUETransformer(LightningModule):
        def __init__(self):
            self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)

        def forward(self, **inputs):
            return self.model(**inputs)


    model = GLUETransformer(model_name_or_path="Intel/bert-base-uncased-mrpc")

The fine-tuned model from Intel could be downloaded from `Intel Hugging Face repository <https://huggingface.co/Intel>`_.

Accuracy-driven quantization config
===================================

Intel® Neural Compressor supports accuracy-driven automatic tuning to generate the optimal INT8 model which meets a predefined accuracy goal. The default tolerance of accuracy loss in the accuracy criterion is 0.01. And the maximum trial number of quantization is 600. The user can specifically define their own criteria by:

.. code-block:: python

    from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion

    accuracy_criterion = AccuracyCriterion(tolerable_loss=0.01)
    tuning_criterion = TuningCriterion(max_trials=600)
    conf = PostTrainingQuantConfig(
        approach="static", backend="default", tuning_criterion=tuning_criterion, accuracy_criterion=accuracy_criterion
    )

The "approach" parameter in PostTrainingQuantConfig is defined by the user to make a choice from post-training static quantization and post-training dynamic by "static" or "dynamic".

Quantize the model
==================

The model can be qutized by Intel® Neural Compressor with:

.. code-block:: python

    from neural_compressor.quantization import fit

    q_model = fit(model=model.model, conf=conf, calib_dataloader=val_dataloader(), eval_func=eval_func)

Users can define the evaluation function "eval_func" by themselves.

At last, the quantized model can be saved by:

.. code-block:: python

    q_model.save("./saved_model/")

*****************
Hands-on Examples
*****************

Based on the `given example code <https://lightning.ai/docs/pytorch/2.1.0/notebooks/lightning_examples/text-transformers.html>`_, we show how Intel Neural Compressor conduct model quantization on PyTorch Lightning. We first define the basic config of the quantization process.

.. code-block:: python

    from neural_compressor.quantization import fit as fit
    from neural_compressor.config import PostTrainingQuantConfig


    def eval_func_for_nc(model_n, trainer_n):
        setattr(model, "model", model_n)
        result = trainer_n.validate(model=model, dataloaders=dm.val_dataloader())
        return result[0]["accuracy"]


    def eval_func(model):
        return eval_func_for_nc(model, trainer)


    conf = PostTrainingQuantConfig()
    q_model = fit(model=model.model, conf=conf, calib_dataloader=dm.val_dataloader(), eval_func=eval_func)

    q_model.save("./saved_model/")

We define the evaluation function as:

.. code-block:: python

    def eval_func_for_nc(model_n, trainer_n):
        setattr(model, "model", model_n)
        result = trainer_n.validate(model=model, dataloaders=dm.val_dataloader())
        return result[0]["accuracy"]


    def eval_func(model):
        return eval_func_for_nc(model, trainer)

Following is the performance comparison between FP32 model and INT8 model:


+-------------+-----------------+------------------+
| Info Type   |  Baseline FP32  |  Quantized INT8  |
+=============+=================+==================+
| Accuracy    | 0.8603          | 0.8578           |
+-------------+-----------------+------------------+
| Duration(s) | 5.8973          | 3.5952           |
+-------------+-----------------+------------------+
| Memory(MB)  | 417.73          | 113.28           |
+-------------+-----------------+------------------+


For more model quantization performance, please refer to `our model list <https://github.com/intel/neural-compressor/blob/master/docs/source/validated_model_list.md>`_

*****************
Technical Support
*****************

Welcome to visit Intel® Neural Compressor website at: https://intel.github.io/neural-compressor to find technical support or contribute your code.
