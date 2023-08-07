## Getting Started

- Login to lightning.ai (_optional_) \<-- takes less than a minute.  ⏩
- Store your models on the cloud \<-- simple call: `upload_model(...)`. 🗳️
- Share it with your friends \<-- just share the "username/model_name" (and version if required) format. :handshake:
- They download using a simple call: `download_model("username/model_name", version="your_version")`. :wink:
- Lightning :zap: fast, isn't it?. :heart:

## Usage

**Storing to the cloud**

```python
import lightning as L

# Upload a checkpoint:
L.store.upload_model("mnist_model", "mnist_model.ckpt")

# Optionally provide a version:
L.store.upload_model("mnist_model", "mnist_model.ckpt", version="1.0.0")
```

**List your models**

```python
import lightning as L

models = L.store.list_models()

print([model.name for model in models])
# ['username/mnist_model']
```

**Downloading from the cloud**

```python
import lightning as L

# Download a checkpoint
L.store.download_model("username/mnist_model", "any_path.ckpt")
```
