################################
Accelerate your code with Fabric
################################


.. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning/fabric/animations/accelerators.mp4
    :width: 800
    :autoplay:
    :loop:
    :muted:
    :nocontrols:


***************************
Set accelerator and devices
***************************

Fabric enables you to take full advantage of the hardware on your system. It supports

- CPU
- GPU (NVIDIA, AMD, Apple Silicon)
- TPU

By default, Fabric tries to maximize the hardware utilization of your system

.. code-block:: python

    # Default settings
    fabric = Fabric(accelerator="auto", devices="auto", strategy="auto")

    # Same as
    fabric = Fabric()

This is the most flexible option and makes your code run on most systems.
You can also explicitly set which accelerator to use:

.. code-block:: python

    # CPU (slow)
    fabric = Fabric(accelerator="cpu")

    # GPU
    fabric = Fabric(accelerator="gpu", devices=1)

    # GPU (multiple)
    fabric = Fabric(accelerator="gpu", devices=8)

    # GPU: Apple M1/M2 only
    fabric = Fabric(accelerator="mps")

    # GPU: NVIDIA CUDA only
    fabric = Fabric(accelerator="cuda", devices=8)

    # TPU
    fabric = Fabric(accelerator="tpu", devices=8)


For running on multiple devices in parallel, also known as "distributed", read our guide for :doc:`Launching Multiple Processes <./launch>`.


----


*****************
Access the Device
*****************

You can access the device anytime through ``fabric.device``.
This lets you replace boilerplate code like this:

.. code-block:: diff

    - if torch.cuda.is_available():
    -     device = torch.device("cuda")
    - else:
    -     device = torch.device("cpu")

    + device = fabric.device
