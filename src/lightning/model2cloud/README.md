## Getting Started

- Login to lightning.ai (_optional_) \<-- takes less than a minute.  ⏩
- Store your models on the cloud \<-- simple call: `to_lightning_cloud(...)`. 🗳️
- Share it with your friends \<-- just share the "username/model_name" (and version if required) format. :handshake:
- They download using a simple call: `download_from_lightning_cloud("username/model_name", version="your_version")`. :wink:
- They load your cool model. `load_from_lightning_cloud("username/model_name", version="your_version")`. :tada:
- Lightning :zap: fast, isn't it?. :heart:

## Usage

**Storing to the cloud**

```python
from lightning.model2cloud import to_lightning_cloud
from sample.model import LitAutoEncoder, Encoder, Decoder

# Initialize your model here
autoencoder = LitAutoEncoder(Encoder(), Decoder())

# Pass the model object:
# No need to pass the username (we'll deduce ourselves), just pass the model name you want as the first argument (with an optional version):
# format: `model_name:version` (version can either be latest or combination of digits and full-stops: 1.0.0 for example)
to_lightning_cloud("unique_model_mnist", model=autoencoder, source_code_path="sample")

# version:
to_lightning_cloud(
    "unique_model_mnist",
    version="1.0.0",
    model=autoencoder,
    source_code_path="sample/model.py",
)

# OR: (this will save the file which has the model defined)
to_lightning_cloud("krshrimali/unique_model_mnist", model=autoencoder)
```

You can also pass the checkpoint path: `to_lightning_cloud("model_name", version="latest", checkpoint_path=...)`.

**Downloading from the cloud**

```python
from lightning.model2cloud import download_from_lightning_cloud

download_from_lightning_cloud("krshrimali/unique_model_mnist", output_dir="your_output_dir")
# OR: (default to lightning_model_storage $HOME/.lightning/lightning_model_store/username/<model_name>/version_<version_with_dots_replaced_by_underscores>/ folder)
download_from_lightning_cloud("krshrimali/unique_model_mnist")
```

**Loading model**

```python
from lightning.model2cloud import load_from_lightning_cloud

# from <username>.<model_name>.version_<version_with_dots_replaced_by_underscores>.<model_source_file> import LitAutoEncoder, Encoder, Decoder
model = load_from_lightning_cloud(
    "<username>/<model_name>>", version="version"
)  # version is optional (defaults to latest)

# OR: load weights or checkpoint (if they were uploaded)
load_from_lightning_cloud(
    "<username>/<model_name>", version="version", load_weights=True / False, load_checkpoint=True / False
)
print(model)
```

**Loading model weights**

```python
from lightning.model2cloud import load_from_lightning_cloud

# If you had passed an `output_dir=...` to download_from_lightning_cloud(...), then you can just do:
# from output_dir.<model_source_file> import LitAutoEncoder, Encoder, Decoder

model = LitAutoEncoder(Encoder(), Decoder())

model = load_from_lightning_cloud(load_weights=True, model=model)
print("State dict: ", model.state_dict())
```

Loading checkpoint is similar, just do: `load_checkpoint=True`.
