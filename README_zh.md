<div align="center">

<img alt="Lightning" src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/ptl_banner.png" width="800px" style="max-width: 100%;">

<br/>
<br/>

**用於預訓練、微調和部署 AI 模型的深度學習框架。**

**NEW- 需要部屬模型嗎? 試試看 [LitServe](https://github.com/Lightning-AI/litserve), 用於模型服務的 PyTorch Lightning**

______________________________________________________________________

<p align="center">
    <a href="#快速開始" style="margin: 0 10px;">快速開始</a> •
  <a href="#範例">範例</a> •
  <a href="#為何使用-pytorch-lightning">PyTorch Lightning</a> •
  <a href="#lightning-fabric-進階控制">Fabric</a> •
  <a href="https://lightning.ai/">Lightning AI</a> •
  <a href="#社群">社群</a> •
  <a href="https://pytorch-lightning.readthedocs.io/en/stable/">文件</a>
</p>

<!-- DO NOT ADD CONDA DOWNLOADS... README CHANGES MUST BE APPROVED BY EDEN OR WILL -->

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-lightning)](https://pypi.org/project/pytorch-lightning/)
[![PyPI Status](https://badge.fury.io/py/pytorch-lightning.svg)](https://badge.fury.io/py/pytorch-lightning)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pytorch-lightning)](https://pepy.tech/project/pytorch-lightning)
[![Conda](https://img.shields.io/conda/v/conda-forge/lightning?label=conda&color=success)](https://anaconda.org/conda-forge/lightning)
[![codecov](https://codecov.io/gh/Lightning-AI/pytorch-lightning/graph/badge.svg?token=SmzX8mnKlA)](https://codecov.io/gh/Lightning-AI/pytorch-lightning)

[![Discord](https://img.shields.io/discord/1077906959069626439?style=plastic)](https://discord.gg/VptPCZkGNa)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/lightning-ai/lightning)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/pytorch-lightning/blob/master/LICENSE)

<!--
[![CodeFactor](https://www.codefactor.io/repository/github/Lightning-AI/lightning/badge)](https://www.codefactor.io/repository/github/Lightning-AI/lightning)
-->

</div>

<div align="center">

<p align="center">

&#160;

<a target="_blank" href="https://lightning.ai/docs/pytorch/latest/starter/introduction.html#define-a-lightningmodule">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/get-started-badge.svg" height="36px" alt="Get started"/>
</a>

</p>

</div>

&#160;

# 為何選擇使用 PyTorch Lightning?

在純 PyTorch 中訓練模型既繁瑣又容易出錯 —— 你必須手動處理反向傳播、混合精度、多 GPU 以及分散式訓練，而且往往每個新專案都需要重寫程式碼。
PyTorch Lightning 將 PyTorch 程式碼進行結構化，幫你自動化這些複雜的部分，讓你能專注於模型和資料，同時保有完整的掌控權，
並且能從 CPU 無縫擴展到多節點，而不需更改核心程式碼。不過，如果你希望自己掌控這些細節，也依然可以選擇更「DIY」的方式。

有趣的比喻：如果說 PyTorch 是 JavaScript，那麼 PyTorch Lightning 就是 ReactJS 或 NextJS。

# Lightning 有兩個核心模組

[PyTorch Lightning: 輕鬆擴展 PyTorch 的訓練及部屬](#%E7%82%BA%E4%BD%95%E4%BD%BF%E7%94%A8-pytorch-lightning).
<br/>
[Lightning Fabric: 進階控制](#lightning-fabric-%E9%80%B2%E9%9A%8E%E6%8E%A7%E5%88%B6).

Lightning 讓你有更精細的控制權，決定你想在 Pytorch 之上新增多少抽象層。

<div align="center">
    <img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/continuum.png" width="80%">
</div>

&#160;

# 快速開始

安裝 Lightning:

```bash
pip install lightning
```

<!-- following section will be skipped from PyPI description -->

<details>
  <summary>進階安裝選項</summary>
    <!-- following section will be skipped from PyPI description -->

#### 安裝額外依賴套件

```bash
pip install lightning['extra']
```

#### Conda

```bash
conda install lightning -c conda-forge
```

#### 安裝穩定版本

從原始碼安裝未來的發行版本
Install future release from the source

```bash
pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/release/stable.zip -U
```

#### 安裝最新開發粄

從原始碼安裝 nightly build 的開發版本 (不保證穩定性)

```bash
pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/master.zip -U
```

或是從 testing PyPI 安裝

```bash
pip install -iU https://test.pypi.org/simple/ pytorch-lightning
```

</details>
<!-- end skipping PyPI description -->

### PyTorch Lightning 範例

定義訓練的流程。這邊是一個簡單的訓練範例 ([探索更多範例](https://lightning.ai/lightning-ai/studios?view=public&section=featured&query=pytorch+lightning)):

```python
# main.py
# ! pip install torchvision
import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
import lightning as L


# --------------------------------
# 步驟一: 定義一個 LightningModule
# --------------------------------
# 一個 LightningModule (nn.Module 的子類別) 定義了一個完整的 *系統*
# (譬如: 一個 LLM, 擴散模型, 自動編碼器, 或是簡單的影像分類器).


class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        # 在 lightning中，forward 定義了預測/推論的動作
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step 定義了訓練的流程。它是獨立於 forward 的
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# -------------------
# 步驟二: 定義資料集
# -------------------
dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())
train, val = data.random_split(dataset, [55000, 5000])

# -------------------
# 步驟三: 開始訓練
# -------------------
autoencoder = LitAutoEncoder()
trainer = L.Trainer()
trainer.fit(autoencoder, data.DataLoader(train), data.DataLoader(val))
```

在你的終端機中執行:

```bash
pip install torchvision
python main.py
```

&#160;

# 為何使用 PyTorch Lightning?

PyTorch Lightning 是一個結構化的 PyTorch 訓練框架 - Lightning 將 PyTorch 程式碼組織起來，並將科學實作及工程內容分離。

![PT to PL](docs/source-pytorch/_static/images/general/pl_quick_start_full_compressed.gif)

&#160;

______________________________________________________________________

### 範例

探索更多不同利用 PyTorch Lightning 進行訓練的範例。預訓練及微調任何模型來執行任何任務，像是分類、分割、摘要等等:

| 任務                                                                                                                | 敘述                                                           | 執行                                                                                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Hello world](#hello-simple-model)                                                                                  | Pretrain - Hello world example                                 | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/pytorch-lightning-hello-world"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a>                  |
| [Image classification](https://lightning.ai/lightning-ai/studios/image-classification-with-pytorch-lightning)       | Finetune - ResNet-34 model to classify images of cars          | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/image-classification-with-pytorch-lightning"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a>    |
| [Image segmentation](https://lightning.ai/lightning-ai/studios/image-segmentation-with-pytorch-lightning)           | Finetune - ResNet-50 model to segment images                   | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/image-segmentation-with-pytorch-lightning"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a>      |
| [Object detection](https://lightning.ai/lightning-ai/studios/object-detection-with-pytorch-lightning)               | Finetune - Faster R-CNN model to detect objects                | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/object-detection-with-pytorch-lightning"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a>        |
| [Text classification](https://lightning.ai/lightning-ai/studios/text-classification-with-pytorch-lightning)         | Finetune - text classifier (BERT model)                        | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/text-classification-with-pytorch-lightning"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a>     |
| [Text summarization](https://lightning.ai/lightning-ai/studios/text-summarization-with-pytorch-lightning)           | Finetune - text summarization (Hugging Face transformer model) | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/text-summarization-with-pytorch-lightning"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a>      |
| [Audio generation](https://lightning.ai/lightning-ai/studios/finetune-a-personal-ai-music-generator)                | Finetune - audio generator (transformer model)                 | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/finetune-a-personal-ai-music-generator"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a>         |
| [LLM finetuning](https://lightning.ai/lightning-ai/studios/finetune-an-llm-with-pytorch-lightning)                  | Finetune - LLM (Meta Llama 3.1 8B)                             | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/finetune-an-llm-with-pytorch-lightning"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a>         |
| [Image generation](https://lightning.ai/lightning-ai/studios/train-a-diffusion-model-with-pytorch-lightning)        | Pretrain - Image generator (diffusion model)                   | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/train-a-diffusion-model-with-pytorch-lightning"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a> |
| [Recommendation system](https://lightning.ai/lightning-ai/studios/recommendation-system-with-pytorch-lightning)     | Train - recommendation system (factorization and embedding)    | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/recommendation-system-with-pytorch-lightning"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a>   |
| [Time-series forecasting](https://lightning.ai/lightning-ai/studios/time-series-forecasting-with-pytorch-lightning) | Train - Time-series forecasting with LSTM                      | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/time-series-forecasting-with-pytorch-lightning"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a> |

______________________________________________________________________

## 進階功能

Lightning 有超過 [40+ 個進階功能](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags)
專為各種規模的專業 AI 研究所設計。

以下是一些範例:

<div align="center">
    <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/features_2.jpg" max-height="600px">
  </div>

<details>
  <summary>在1000個GPU下進行訓練</summary>

```python
# 8 GPUs
# no code changes needed
trainer = Trainer(accelerator="gpu", devices=8)

# 256 GPUs
trainer = Trainer(accelerator="gpu", devices=8, num_nodes=32)
```

</details>

<details>
  <summary>在其他加速器下（如TPU）下進行訓練</summary>

```python
# no code changes needed
trainer = Trainer(accelerator="tpu", devices=8)
```

</details>

<details>
  <summary>16-bit 浮點精度訓練</summary>

```python
# no code changes needed
trainer = Trainer(precision=16)
```

</details>

<details>
  <summary>進行實驗數據管理</summary>

```python
from lightning import loggers

# tensorboard
trainer = Trainer(logger=TensorBoardLogger("logs/"))

# weights and biases
trainer = Trainer(logger=loggers.WandbLogger())

# comet
trainer = Trainer(logger=loggers.CometLogger())

# mlflow
trainer = Trainer(logger=loggers.MLFlowLogger())

# neptune
trainer = Trainer(logger=loggers.NeptuneLogger())

# ... and dozens more
```

</details>

<details>

<summary>Early Stopping</summary>

```python
es = EarlyStopping(monitor="val_loss")
trainer = Trainer(callbacks=[es])
```

</details>

<details>
  <summary>Checkpointing</summary>

```python
checkpointing = ModelCheckpoint(monitor="val_loss")
trainer = Trainer(callbacks=[checkpointing])
```

</details>

<details>
  <summary>模型輸出至 torchscript (JIT) 格式 (用於生產環境)</summary>

```python
# torchscript
autoencoder = LitAutoEncoder()
torch.jit.save(autoencoder.to_torchscript(), "model.pt")
```

</details>

<details>
  <summary>模型輸出至 ONNX 格式 (用於生產環境)</summary>

```python
# onnx
with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmpfile:
    autoencoder = LitAutoEncoder()
    input_sample = torch.randn((1, 64))
    autoencoder.to_onnx(tmpfile.name, input_sample, export_params=True)
    os.path.isfile(tmpfile.name)
```

</details>

______________________________________________________________________

## 相比原始未結構化的 PyTorch 的優點

- 模型變得更與硬體底層脫鉤
- 程式碼更易讀，因為工程代碼被抽象化
- 更容易複驗
- 因為 Lightning 處理了棘手的工程問題，所以犯錯的機會更少
- 仍然保持著所有的彈性（LightningModules 仍然是 PyTorch 模組），但去除了大量重複性代碼
- Lightning 與數十種流行的機器學習工具整合
- [每個新的 PR 都經過嚴格測試](https://github.com/Lightning-AI/lightning/tree/master/tests)。我們測試了各種組合的 PyTorch 和 Python 支持版本、每個操作系統、多 GPU 甚至 TPU。
- 最小的運行速度開銷（與純 PyTorch 相比，每個 epoch 約增加 300 毫秒）。

______________________________________________________________________

<div align="center">
    <a href="https://lightning.ai/docs/pytorch/stable/">閱讀 PyTorch Lightning 文件</a>
</div>

______________________________________________________________________

&#160;
&#160;

# Lightning Fabric: 進階控制

在任何硬體設備、任何規模下對 PyTorch 訓練迴圈和擴展策略進行專家級的控制。你甚至可以自己撰寫 Trainer。

Fabric 是設計給在各種硬體規模下訓練各種最為複雜的模型、例如 founding model scaling、LLMs、diffusion、transformers、強化學習及主動學習。

<table>
<tr>
<th>調整部分</th>
<th>使用 Fabric 的結果 (copy me!)</th>
</tr>
<tr>
<td>
<sub>

```diff
+ import lightning as L
  import torch; import torchvision as tv

 dataset = tv.datasets.CIFAR10("data", download=True,
                               train=True,
                               transform=tv.transforms.ToTensor())

+ fabric = L.Fabric()
+ fabric.launch()

  model = tv.models.resnet18()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
- device = "cuda" if torch.cuda.is_available() else "cpu"
- model.to(device)
+ model, optimizer = fabric.setup(model, optimizer)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
+ dataloader = fabric.setup_dataloaders(dataloader)

  model.train()
  num_epochs = 10
  for epoch in range(num_epochs):
      for batch in dataloader:
          inputs, labels = batch
-         inputs, labels = inputs.to(device), labels.to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = torch.nn.functional.cross_entropy(outputs, labels)
-         loss.backward()
+         fabric.backward(loss)
          optimizer.step()
          print(loss.data)
```

</sub>
<td>
<sub>

```Python
import lightning as L
import torch;
import torchvision as tv

dataset = tv.datasets.CIFAR10("data", download=True,
                              train=True,
                              transform=tv.transforms.ToTensor())

fabric = L.Fabric()
fabric.launch()

model = tv.models.resnet18()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
model, optimizer = fabric.setup(model, optimizer)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
dataloader = fabric.setup_dataloaders(dataloader)

model.train()
num_epochs = 10
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        fabric.backward(loss)
        optimizer.step()
        print(loss.data)
```

</sub>
</td>
</tr>
</table>

## 重點功能

<details>
  <summary>輕易地在 CPU 與 GPU (Apple Silicon, CUDA, …), TPU, 多 GPU 或甚至是多節點訓練模式下進行切換</summary>

```python
# 使用你既有的硬體
# 不須修改程式碼
fabric = Fabric()

# 在 GPUs 上執行 (CUDA 或 MPS)
fabric = Fabric(accelerator="gpu")

# 使用 8 張 GPUs
fabric = Fabric(accelerator="gpu", devices=8)

# 使用 256 張 GPUs，跨 32 個節點
fabric = Fabric(accelerator="gpu", devices=8, num_nodes=32)

# 在 TPU 上執行
fabric = Fabric(accelerator="tpu")
```

</details>

<details>
  <summary>開箱及用任何最先進的分散式訓練策略（DDP、FSDP、DeepSpeed）及混精度訓練。</summary>

```python
# 使用任何分散式訓練策略
fabric = Fabric(strategy="ddp")
fabric = Fabric(strategy="deepspeed")
fabric = Fabric(strategy="fsdp")

# 進行訓練精度切換
fabric = Fabric(precision="16-mixed")
fabric = Fabric(precision="64")
```

</details>

<details>
  <summary>所有與硬體相關的重複性代碼都已經為你處理好</summary>

```diff
  # 再也不會出現這種東西!
- model.to(device)
- batch.to(device)
```

</details>

<details>
  <summary>利用 Fabric 元件建構自定義的 Trainer 來支援 checkpointing、訓練日誌及其他功能</summary>

```python
import lightning as L


class MyCustomTrainer:
    def __init__(self, accelerator="auto", strategy="auto", devices="auto", precision="32-true"):
        self.fabric = L.Fabric(accelerator=accelerator, strategy=strategy, devices=devices, precision=precision)

    def fit(self, model, optimizer, dataloader, max_epochs):
        self.fabric.launch()

        model, optimizer = self.fabric.setup(model, optimizer)
        dataloader = self.fabric.setup_dataloaders(dataloader)
        model.train()

        for epoch in range(max_epochs):
            for batch in dataloader:
                input, target = batch
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                self.fabric.backward(loss)
                optimizer.step()
```

你可以在我們的[範例](examples/fabric/build_your_own_trainer)中找到更完整的範例

</details>

______________________________________________________________________

<div align="center">
    <a href="https://lightning.ai/docs/fabric/stable/">閱讀 Lightning Fabric 文件</a>
</div>

______________________________________________________________________

&#160;
&#160;

## 範例

###### 自監督式學習

- [CPC transforms](https://lightning-bolts.readthedocs.io/en/stable/transforms/self_supervised.html#cpc-transforms)
- [Moco v2 transforms](https://lightning-bolts.readthedocs.io/en/stable/transforms/self_supervised.html#moco-v2-transforms)
- [SimCLR transforms](https://lightning-bolts.readthedocs.io/en/stable/transforms/self_supervised.html#simclr-transforms)

###### 卷積架構

- [GPT-2](https://lightning-bolts.readthedocs.io/en/stable/models/convolutional.html#gpt-2)
- [UNet](https://lightning-bolts.readthedocs.io/en/stable/models/convolutional.html#unet)

###### 強化學習

- [DQN Loss](https://lightning-bolts.readthedocs.io/en/stable/losses.html#dqn-loss)
- [Double DQN Loss](https://lightning-bolts.readthedocs.io/en/stable/losses.html#double-dqn-loss)
- [Per DQN Loss](https://lightning-bolts.readthedocs.io/en/stable/losses.html#per-dqn-loss)

###### GANs

- [Basic GAN](https://lightning-bolts.readthedocs.io/en/stable/models/gans.html#basic-gan)
- [DCGAN](https://lightning-bolts.readthedocs.io/en/stable/models/gans.html#dcgan)

###### 傳統 ML

- [Logistic Regression](https://lightning-bolts.readthedocs.io/en/stable/models/classic_ml.html#logistic-regression)
- [Linear Regression](https://lightning-bolts.readthedocs.io/en/stable/models/classic_ml.html#linear-regression)

&#160;
&#160;

## Continuous Integration

Lightning 經過了在各種組合的 PyTorch 和 Python 支持版本、每個操作系統、多 GPU 甚至 TPU下的嚴格測試。

###### \*Codecov is > 90%+ but build delays may show less

<details>
  <summary>當前建置結果</summary>

<center>

|       System / PyTorch ver.        |                                                                                              1.13                                                                                               |                                                                                               2.0                                                                                               |                                                                                                                 2.1                                                                                                                 |
| :--------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|         Linux py3.9 [GPUs]         |                                                                                                                                                                                                 |                                                                                                                                                                                                 | [![Build Status](https://dev.azure.com/Lightning-AI/lightning/_apis/build/status%2Fpytorch-lightning%20%28GPUs%29?branchName=master)](https://dev.azure.com/Lightning-AI/lightning/_build/latest?definitionId=24&branchName=master) |
|  Linux (multiple Python versions)  | [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml) | [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml) |                   [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml)                   |
|   OSX (multiple Python versions)   | [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml) | [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml) |                   [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml)                   |
| Windows (multiple Python versions) | [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml) | [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml) |                   [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml)                   |

</center>
</details>

&#160;
&#160;

## 社群

Lightning 的社群是由以下成員維護

- [10+ 核心貢獻者](https://lightning.ai/docs/pytorch/latest/community/governance.html)，包含多位來自各頂尖AI研究組織的專業工程師，科學家及博士生。
- 800+ 社群貢獻者.

想要來協助我們打造 Lightning，並為數千名研究人員減少重複性代碼嗎？[點此了解如何進行你的第一次貢獻](https://lightning.ai/docs/pytorch/stable/generated/CONTRIBUTING.html)

Lightning 同時也屬於 [PyTorch Lightning 生態系統](https://pytorch.org/ecosystem/)，該生態系統要求專案必須有完善的測試、文件及支援。

### 尋求幫助

若您有任何問題，請參考以下資源:

1. [Read the docs](https://lightning.ai/docs).
1. [Search through existing Discussions](https://github.com/Lightning-AI/lightning/discussions),
   or [add a new question](https://github.com/Lightning-AI/lightning/discussions/new)
1. [Join our discord](https://discord.com/invite/tfXFetEZxv).
