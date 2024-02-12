<div align="center">

<img alt="Lightning" src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LightningColor.png" width="800px" style="max-width: 100%;">

<br/>
<br/>

## Blazing fast, distributed streaming of training data from cloud storage

</div>

# Lightning Data

We developed `Streaming Dataset` to optimize training of large datasets from cloud storage, prioritizing speed, affordability, and scalability.

Specifically crafted for multi-node, distributed training with large models, it enhances accuracy, performance, and user-friendliness. Now, training efficiently is possible regardless of the data's location. Simply stream in the required data when needed.

The `Streaming Dataset` is compatible with any data type, including **images, text, video, and multimodal data** and it is a drop-in replacement for your PyTorch [IterableDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) class.

# Lightning Data:

The Lightning Data framework provides several primitives to make data manipulation in the cloud highly optimized.

- The `optimize` operator enables to apply a given python function over a list of inputs and serialized the return or yieled values.
- The `StreamingDataset` and `StreamingDataLoader` provides an efficient mechanism to stream data transformed with the `optimize` operator.
- The `map` operator enables to apply a given python function over a list of inputs and store the files written to the provided output directory.

Go on [lightning.ai](https://lightning.ai) and create a free account to try the examples below.

______________________________________________________________________

# Lightning Data Optimize Operator

______________________________________________________________________

# Lightning Data Map Operator

Lightning Data `map` splits evenly the inputs across workers and machines.

In the example below, the `map` operator is used to resize the ImageNet test set.

### Resize images

```python
import os
from PIL import Image
from lightning_cloud.utils import add_s3_connection
from lightning.data import map

# 1. Add an external S3 bucket containing some data (Imagenet)
add_s3_connection("imagenet-1m-template")

# 2. Define a function to be applied over the inputs.
# Here, we simply read an image, resize it, and write it back.
# Behind the scenes, the input and output data live on s3
# but the map operators will stream the machine to operate on the machine filesystem.
def resize_fn(input_filepath, output_dir):
    img = Image.open(input_filepath).resize((224, 224))
    output_filepath = os.path.join(output_dir, os.path.basename(input_filepath))
    img.save(output_filepath)

# 3. Generate the inputs (we are going to resize Imagenet testing set)
input_dir = "/teamspace/s3_connections/imagenet-1m-template/raw/test"
inputs = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
# inputs = ['/teamspace/s3_connections/imagenet-1m-template/raw/test/ILSVRC2012_test_00000001.JPEG', ...]

# 4. Store the resized images wherever you want
outputs = map(
  resize_fn,
  inputs,
  output_dir="output_dir",
)
```

Run this script on the Studio terminal:

```bash
python main.py
```
