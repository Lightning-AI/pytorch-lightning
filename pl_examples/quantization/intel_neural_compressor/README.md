# Step-by-Step

This document describes the step-by-step instructions for reproducing PyTorch ResNet50/ResNet18/ResNet101 tuning results with [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

# Prerequisite

### 1. Installation

```shell
pip install neural-compressor
pip install torch>=1.9.0 torchvision>=0.10.0 --find-links https://download.pytorch.org/whl/torch_stable.html
```

### 2. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) raw image to the following directory: /path/to/imagenet.  The directory should include a train and val folders:

```bash
ls /path/to/imagenet
train  val
```

# Quantize the model

### Write Yaml Config File

```yaml
model:
  name: imagenet
  framework: pytorch_fx                              # supported values are pytorch_fx, pytorch_ipex here.

device: cpu                                          # Now only support CPU device, will support Intel GPU future.

quantization:
  approach: post_training_static_quant               # supported values are post_training_static_quant, post_training_dynamic_quant, quant_aware_training.
  calibration:
    sampling_size: 300

tuning:
  accuracy_criterion:
    relative:  0.01                                  # optional. default value is relative, other value is absolute. this example allows relative accuracy loss: 1%.
  exit_policy:
    timeout: 0                                       # optional. tuning timeout (seconds). default value is 0 which means early stop. combine with max_trials field to decide when to exit.
    max_trials: 1200
  random_seed: 9527                                  # optional. random seed for deterministic tuning.
```

- For "framework" filed, the "pytorch_fx" is for FX mode, and "pytorch_ipex" is for intel extension of PyTorch.

- For quantization approach, Please refer below link:
  [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)

### Saving and loading model:

- Saving quantization configure and weights:
  After tuning with Neural Compressor, we can get neural_compressor.model:

```
cli = MyLightningCLI(ImageNetLightningModel,
                     seed_everything_default=42,
                     save_config_overwrite=True, run=False)
callback = pl.callbacks.INCQuantization(
    'config/quantization.yaml', monitor="val_acc1", module_name_to_quant="model",
    dirpath=cli.trainer.default_root_dir
)
cli.trainer.callbacks.append(callback)
cli.trainer.fit(cli.model, datamodule=cli.datamodule)
```

The quantized model saved in args.default_root_dir

- Create and quantize the model:

```python
model = ImageNetLightningModel(**vars(args))  # fp32 model
from neural_compressor.utils.pytorch import load

model.model = load(args.default_root_dir, model.model)
```

Please refer to [Sample code](./imagenet.py).

# Run

### 1. ResNet50

```shell

python imagenet.py --model.arch resnet50 --model.pretrained true --model.data_path /path/to/imagenet --model.batch_size 300 --trainer.default_root_dir ./checkpoint
```

> **Note**
>
> batch size should be the number which is sampling size divisible by. sampling size is a setting in config/quantization.yaml

### 2. ResNet18

```shell
python imagenet.py --model.arch resnet18 --model.pretrained true --model.data_path /path/to/imagenet --model.batch_size 300 --trainer.default_root_dir ./checkpoint
```

### 3. ResNext101_32x8d

```shell
python imagenet.py --model.arch resnext101_32x8d --model.pretrained true --model.data_path /path/to/imagenet --model.batch_size 300 --trainer.default_root_dir ./checkpoint
```

### 4. InceptionV3

```shell
python imagenet.py --model.arch inception_v3 --model.pretrained true --model.data_path /path/to/imagenet --model.batch_size 300 --trainer.default_root_dir ./checkpoint
```

# Benchmark

### int8 performance:

python imagenet.py --model.arch resnet50 ---model.pretrained true --model.data_path /path/to/imagenet --model.batch_size 300 --trainer.default_root_dir ./checkpoint -e
