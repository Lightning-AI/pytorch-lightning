## Getting Started

- Login to lightning.ai (_optional_) \<-- takes less than a minute.  ⏩
- Store your models on the cloud \<-- simple call: `upload_model(...)`. 🗳️
- Share it with your friends \<-- just share the "username/model_name" (and version if required) format. :handshake:
- They download using a simple call: `download_model("username/model_name", version="your_version")`. :wink:
- They load your cool model. `load_model("username/model_name", version="your_version")`. :tada:
- Lightning :zap: fast, isn't it?. :heart:

## Usage

**Storing to the cloud**

```python
import lightning as L
from sample.model import LitAutoEncoder, Encoder, Decoder

# Initialize your model here
autoencoder = LitAutoEncoder(Encoder(), Decoder())

# Pass the model object:
# No need to pass the username (we'll deduce ourselves), just pass the model name you want as the first argument (with an optional version):
# format: `model_name:version` (version can either be latest or combination of digits and full-stops: 1.0.0 for example)
L.store.upload_model("unique_model_mnist", model=autoencoder, source_code_path="sample")

# version:
L.store.upload_model(
    "unique_model_mnist",
    version="1.0.0",
    model=autoencoder,
    source_code_path="sample/model.py",
)

# OR: (this will save the file which has the model defined)
L.store.upload_model("krshrimali/unique_model_mnist", model=autoencoder)
```

You can also pass the checkpoint path: `upload_model("model_name", version="latest", checkpoint_path=...)`.

**Downloading from the cloud**

At first, you need to download the model to your local machine.

```python
import lightning as L

L.store.download_model(
    "krshrimali/unique_model_mnist",
    output_dir="your_output_dir",
)
# OR: (default to model_storage
#        $HOME
#         |- .lightning
#         |  |- model_store
#         |  |  |- username
#         |  |  |  |- <model_name>
#         |  |  |  |  |- version_<version_with_dots_replaced_by_underscores>
#      folder)
L.store.download_model("krshrimali/unique_model_mnist")
```

**Loading model**

Then you can load the model to your program.

```python
import lightning as L

# from <username>.<model_name>.version_<version_with_dots_replaced_by_underscores>.<model_source_file> import LitAutoEncoder, Encoder, Decoder
model = L.store.load_model("<username>/<model_name>>", version="version")  # version is optional (defaults to latest)

# OR: load weights or checkpoint (if they were uploaded)
L.store.load_model(
    "<username>/<model_name>", version="version", load_weights=True | False, load_checkpoint=True | False
)
print(model)
```

**Loading model weights**

```python
import lightning as L
from sample.model import LitAutoEncoder, Encoder, Decoder

# If you had passed an `output_dir=...` to download_model(...), then you can just do:
# from output_dir.<model_source_file> import LitAutoEncoder, Encoder, Decoder

model = LitAutoEncoder(Encoder(), Decoder())

model = L.store.load_model(load_weights=True, model=model)
print("State dict: ", model.state_dict())
```

Loading checkpoint is similar, just do: `load_checkpoint=True`.
