## DCGAN

This is an example of a GAN (Generative Adversarial Network) that learns to generate realistic images of faces.
We show two code versions:
The first one is implemented in raw PyTorch, but isn't easy to scale.
The second one is using [Lightning Fabric](https://lightning.ai/docs/fabric) to accelerate and scale the model.

Tip: You can easily inspect the difference between the two files with:

```bash
sdiff train_torch.py train_fabric.py
```

|                                                         Real                                                         |                                                     Generated                                                      |
| :------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------: |
| ![sample-data](https://user-images.githubusercontent.com/5495193/206484557-2e9e3810-a9c8-4ae0-bc6e-126866fef4f0.png) | ![fake-7914](https://user-images.githubusercontent.com/5495193/206484621-5dc4a9a6-c782-4c71-8e80-27580cdcc7e6.png) |

### Run

**Raw PyTorch:**

```bash
python train_torch.py
```

**Accelerated using Lightning Fabric:**

```bash
python train_fabric.py
```

Generated images get saved to the _outputs_ folder.

### Notes

The CelebA dataset is hosted through a Google Drive link by the authors, but the downloads are limited.
You may get a message saying that the daily quota was reached. In this case,
[manually download the data](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg)
through your browser.

### References

- [DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- [Large-scale CelebFaces Attributes (CelebA) Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
