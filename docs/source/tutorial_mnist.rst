MNIST
===============
Although Lightning can support any kind of research, use this as a template
to understand the basic use case with a simple MNIST classifier.

Every research project requires the same core ingredients:
1. A model.
2. Train/val/test data
3. Optimizer(s)
4. Training step computations

PyTorch Lightning does nothing more than organize and structure pure PyTorch code.

IMAGE OF STRUCTURE

The Model
---------
The LightningModule provides the structure on how to organize these 5 ingredients.

Let's first start with the model. In this case we'll design
a 3-layer neural network.

.. code-block:: default

    import torch
    from torch.nn import functional as F
    from torch import nn
    import pytorch_lightning as pl

    class CoolMNIST(pl.LightningModule):

      def __init__(self):
        super(CoolMNIST, self).__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

      def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3
        x = self.layer_3(x)

        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)

        return x

Notice this is a `LightningModule` instead of a `torch.nn.Module`. A LightningModule is
equivalent to a PyTorch Module except it has added functionality. However, you can use it
EXACTLY the same as you would a PyTorch Module.

.. code-block:: default

    net = CoolMNIST()
    x = torch.Tensor(1, 1, 28, 28)
    out = net(x)

.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    torch.Size([1, 10])

Data
----

The Lightning Module organizes your dataloaders and data processing as well.
Here's the PyTorch code for loading MNIST

.. code-block:: default

    from torch.utils.data import DataLoader, random_split
    from torchvision.datasets import MNIST
    import os
    from torchvision import datasets, transforms


    # transforms
    # prepare transforms standard to MNIST
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # data
    mnist_train = MNIST(os.getcwd(), train=True, download=True)
    mnist_train = DataLoader(mnist_train, batch_size=64)

When using PyTorch Lightning, we use the exact same code except we organize it into
the LightningModule

.. code-block:: python

    from torch.utils.data import DataLoader, random_split
    from torchvision.datasets import MNIST
    import os
    from torchvision import datasets, transforms

    class CoolMNIST(pl.LightningModule):

      def train_dataloader(self):
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transform)
        return DataLoader(mnist_train, batch_size=64)

Notice the code is exactly the same, except now the training dataloading has been organized by the LightningModule
under the `train_dataloader` method. This is great because if you run into a project that uses Lightning and want
to figure out how they prepare their training data you can just look in the `train_dataloader` method.

Optimizer
---------
Next we choose what optimizer to use for training our system.
In PyTorch we do it as follows:

.. code-block:: python

    from torch.optim import Adam
    optimizer = Adam(CoolMNIST().parameters(), lr=1e-3)


In Lightning we do the same but organize it under the configure_optimizers method.
If you don't define this, Lightning will automatically use `Adam(self.parameters(), lr=1e-3)`.

.. code-block:: python

    class CoolMNIST(pl.LightningModule):

      def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

Training step
-------------

The training step is what happens inside the training loop.

.. code-block:: python

    for epoch in epochs:
        for batch in data:
            # TRAINING STEP
            # ....
            # TRAINING STEP
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

In the case of MNIST we do the following

.. code-block:: python

    for epoch in epochs:
        for batch in data:
            # TRAINING STEP START
            x, y = batch
            logits = model(x)
            loss = F.nll_loss(logits, x)
            # TRAINING STEP END

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

In Lightning, everything that is in the training step gets organized under the `training_step` function
in the LightningModule

.. code-block:: python

    class CoolMNIST(pl.LightningModule):

      def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, x)
        return {'loss': loss}
        # return loss (also works)

Again, this is the same PyTorch code except that it has been organized by the LightningModule.
This code is not restricted which means it can be as complicated as a full seq-2-seq, RL loop, GAN, etc...

Training
--------
So far we defined 4 key ingredients in pure PyTorch but organized the code inside the LightningModule.

1. Model.
2. Training data.
3. Optimizer.
4. What happens in the training loop.

For clarity, we'll recall that the full LightningModule now looks like this.

.. code-block:: python

    class CoolMNIST(pl.LightningModule):
      def __init__(self):
        super(CoolMNIST, self).__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

      def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.relu(x)
        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)
        return x

      def train_dataloader(self):
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transform)
        return DataLoader(mnist_train, batch_size=64)

      def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

      def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, x)
        return {'loss': loss}

Again, this is the same PyTorch code, except that it's organized
by the LightningModule. This organization now lets us train this model

.. code-block:: python

    from pytorch_lightning import Trainer

    model = CoolMNIST()
    trainer = Trainer()
    trainer.fit(model)

But the beauty is all the magic you can do with the trainer flags. For instance, run this model on a TPU
without changing the code (make sure to change to the TPU runtime on Colab.

.. code-block:: python

    model = CoolMNIST()
    trainer = Trainer(num_tpu_cores=8)
    trainer.fit(model)

Or you can also train on multiple GPUs

.. code-block:: python

    model = CoolMNIST()
    trainer = Trainer(gpus=1)
    trainer.fit(model)

Or multiple nodes

.. code-block:: python

    # (32 GPUs)
    model = CoolMNIST()
    trainer = Trainer(gpus=8, num_nodes=4, distributed_backend='ddp')
    trainer.fit(model)







Now we can train the LightningModule without doing anything else!

.. code-block:: python

    model = CoolMNIST()
    trainer = pl.Trainer()
    trainer.fit()



Neural networks can be constructed using the ``torch.nn`` package.

Now that you had a glimpse of ``autograd``, ``nn`` depends on
``autograd`` to define models and differentiate them.
An ``nn.Module`` contains layers, and a method ``forward(input)``\ that
returns the ``output``.

For example, look at this network that classifies digit images:

.. figure:: /_static/img/mnist.png
   :alt: convnet

   convnet

It is a simple feed-forward network. It takes the input, feeds it
through several layers one after the other, and then finally gives the
output.

A typical training procedure for a neural network is as follows:

- Define the neural network that has some learnable parameters (or
  weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using a simple update rule:
  ``weight = weight - learning_rate * gradient``

Define the network
------------------

Let’s define this network:


.. code-block:: default

    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            # 1 input image channel, 6 output channels, 3x3 square convolution
            # kernel
            self.conv1 = nn.Conv2d(1, 6, 3)
            self.conv2 = nn.Conv2d(6, 16, 3)
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            # Max pooling over a (2, 2) window
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            # If the size is a square you can only specify a single number
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def num_flat_features(self, x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features


    net = Net()
    print(net)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Net(
      (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
      (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
      (fc1): Linear(in_features=576, out_features=120, bias=True)
      (fc2): Linear(in_features=120, out_features=84, bias=True)
      (fc3): Linear(in_features=84, out_features=10, bias=True)
    )


You just have to define the ``forward`` function, and the ``backward``
function (where gradients are computed) is automatically defined for you
using ``autograd``.
You can use any of the Tensor operations in the ``forward`` function.

The learnable parameters of a model are returned by ``net.parameters()``


.. code-block:: default


    params = list(net.parameters())
    print(len(params))
    print(params[0].size())  # conv1's .weight





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    10
    torch.Size([6, 1, 3, 3])


Let's try a random 32x32 input.
Note: expected input size of this net (LeNet) is 32x32. To use this net on
the MNIST dataset, please resize the images from the dataset to 32x32.


.. code-block:: default


    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    tensor([[ 0.0127, -0.0025, -0.0628, -0.1181, -0.0699, -0.1076,  0.0286,  0.0172,
             -0.0834,  0.1178]], grad_fn=<AddmmBackward>)


Zero the gradient buffers of all parameters and backprops with random
gradients:


.. code-block:: default

    net.zero_grad()
    out.backward(torch.randn(1, 10))







.. note::

    ``torch.nn`` only supports mini-batches. The entire ``torch.nn``
    package only supports inputs that are a mini-batch of samples, and not
    a single sample.

    For example, ``nn.Conv2d`` will take in a 4D Tensor of
    ``nSamples x nChannels x Height x Width``.

    If you have a single sample, just use ``input.unsqueeze(0)`` to add
    a fake batch dimension.

Before proceeding further, let's recap all the classes you’ve seen so far.

**Recap:**
  -  ``torch.Tensor`` - A *multi-dimensional array* with support for autograd
     operations like ``backward()``. Also *holds the gradient* w.r.t. the
     tensor.
  -  ``nn.Module`` - Neural network module. *Convenient way of
     encapsulating parameters*, with helpers for moving them to GPU,
     exporting, loading, etc.
  -  ``nn.Parameter`` - A kind of Tensor, that is *automatically
     registered as a parameter when assigned as an attribute to a*
     ``Module``.
  -  ``autograd.Function`` - Implements *forward and backward definitions
     of an autograd operation*. Every ``Tensor`` operation creates at
     least a single ``Function`` node that connects to functions that
     created a ``Tensor`` and *encodes its history*.

**At this point, we covered:**
  -  Defining a neural network
  -  Processing inputs and calling backward

**Still Left:**
  -  Computing the loss
  -  Updating the weights of the network

Loss Function
-------------
A loss function takes the (output, target) pair of inputs, and computes a
value that estimates how far away the output is from the target.

There are several different
`loss functions <https://pytorch.org/docs/nn.html#loss-functions>`_ under the
nn package .
A simple loss is: ``nn.MSELoss`` which computes the mean-squared error
between the input and the target.

For example:


.. code-block:: default


    output = net(input)
    target = torch.randn(10)  # a dummy target, for example
    target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()

    loss = criterion(output, target)
    print(loss)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    tensor(0.7406, grad_fn=<MseLossBackward>)


Now, if you follow ``loss`` in the backward direction, using its
``.grad_fn`` attribute, you will see a graph of computations that looks
like this:

::

    input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
          -> view -> linear -> relu -> linear -> relu -> linear
          -> MSELoss
          -> loss

So, when we call ``loss.backward()``, the whole graph is differentiated
w.r.t. the loss, and all Tensors in the graph that has ``requires_grad=True``
will have their ``.grad`` Tensor accumulated with the gradient.

For illustration, let us follow a few steps backward:


.. code-block:: default


    print(loss.grad_fn)  # MSELoss
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    <MseLossBackward object at 0x7fbd35b2cc88>
    <AddmmBackward object at 0x7fbd35b2cd30>
    <AccumulateGrad object at 0x7fbd35b2cd30>


Backprop
--------
To backpropagate the error all we have to do is to ``loss.backward()``.
You need to clear the existing gradients though, else gradients will be
accumulated to existing gradients.


Now we shall call ``loss.backward()``, and have a look at conv1's bias
gradients before and after the backward.


.. code-block:: default



    net.zero_grad()     # zeroes the gradient buffers of all parameters

    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    loss.backward()

    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    conv1.bias.grad before backward
    tensor([0., 0., 0., 0., 0., 0.])
    conv1.bias.grad after backward
    tensor([ 0.0038, -0.0053,  0.0007, -0.0004,  0.0054,  0.0005])


Now, we have seen how to use loss functions.

**Read Later:**

  The neural network package contains various modules and loss functions
  that form the building blocks of deep neural networks. A full list with
  documentation is `here <https://pytorch.org/docs/nn>`_.

**The only thing left to learn is:**

  - Updating the weights of the network

Update the weights
------------------
The simplest update rule used in practice is the Stochastic Gradient
Descent (SGD):

     ``weight = weight - learning_rate * gradient``

We can implement this using simple Python code:

.. code:: python

    learning_rate = 0.01
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)

However, as you use neural networks, you want to use various different
update rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc.
To enable this, we built a small package: ``torch.optim`` that
implements all these methods. Using it is very simple:


.. code-block:: default


    import torch.optim as optim

    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()    # Does the update








.. Note::

      Observe how gradient buffers had to be manually set to zero using
      ``optimizer.zero_grad()``. This is because gradients are accumulated
      as explained in the `Backprop`_ section.


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  3.783 seconds)


.. _sphx_glr_download_beginner_blitz_neural_networks_tutorial.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: neural_networks_tutorial.py <neural_networks_tutorial.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: neural_networks_tutorial.ipynb <neural_networks_tutorial.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_