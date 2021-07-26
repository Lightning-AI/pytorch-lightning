.. _grid:

##############
Cloud Training
##############

Lightning makes it easy to scale your training, without the boilerplate.
If you want to train your models on the cloud, without dealing with engineering infrastructure and servers, you can try `Grid.ai <https://www.grid.ai/>`_.

Developed by the creators of `PyTorch Lightning <https://www.pytorchlightning.ai/>`_, Grid is a platform that allows you to:


- **Scale your models to multi-GPU and multiple nodes** instantly with interactive sessions
- **Run Hyperparameter Sweeps on 100s of GPUs** in one command
- **Upload huge datasets** for availability at scale
- **Iterate faster and cheaper**, you only pay for what you need


****************
Training on Grid
****************

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://grid-docs.s3.us-east-2.amazonaws.com/grid.png"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/grid.mp4"></video>

|

You can launch any Lightning model on Grid using `grid run` `CLI <https://pypi.org/project/lightning-grid/>`_:

.. code-block:: bash

    grid run --instance_type v100 --gpus 4 my_model.py --gpus 4 --learning_rate 'uniform(1e-6, 1e-1, 20)' --layers '[2, 4, 8, 16]'

You can also start runs or interactive sessions from the `Grid platform <https://platform.grid.ai>`_, where you can upload datasets, view artifacts, view the logs, the cost, log into tensorboard, and so much more.


**********
Learn More
**********

`Sign up for Grid <http://platform.grid.ai>`_ and receive free credits to get you started!

`Grid in 3 minutes <https://docs.grid.ai/#introduction>`_

`Grid.ai Terms of Service <https://www.grid.ai/terms-of-service/>`_

