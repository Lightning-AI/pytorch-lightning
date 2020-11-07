## Hydra Pytorch Lightning Example

This directory consists of an example of configuring Pytorch Lightning with [Hydra](https://hydra.cc/). Hydra is a tool that allows for the easy configuration of complex applications.  
The core of this directory consists of a set of structured configs used for pytorch lightining, which are stored under the `from pytorch_lightning.trainer.trainer_conf import PLConfig`. Within the PL config there are 5 cofigurations: 1) Trainer Configuration, 2) Profiler Configuration, 3) Early Stopping Configuration, 4) Logger Configuration and 5) Checkpoint Configuration. All of these are basically mirrors of the arguments that make up these objects. These configuration are used to instantiate the objects using Hydras instantiation utility.

Aside from the PyTorch Lightning configuration we have included a few other important configurations. Optimizer and Scheduler are easy off-the-shelf configurations for configuring your optimizer and learning rate scheduler. You can add them to your config defaults list as needed and use them to configure these objects. Additionally, we provide the arch and data configurations for changing model and data hyperparameters.

All of the above hyperparameters are configured in the config.yaml file which contains the top level configuration for all these configurations. Under this file is a defaults list which highlights for each of these Hydra groups what is the default configuration. Beyond this configuration file, all of the parameters defined can be overriden via the command line.

Additionally, for type safety we highlight in our file `user_config.py` an example of extending the `PLConfig` data class with a user configuration. Hence, we can get the benefits of type safety for our entire config.yaml. Please read through the [basic tutorial](https://hydra.cc/docs/next/tutorials/basic/your_first_app/simple_cli) and [structured configuration tutorial](https://hydra.cc/docs/next/tutorials/structured_config/intro) for more information on using Hydra.

### Tensorboard Visualization

Hydra by default changes the running directory of your program when running into outputs/[DATE]/[TIME]. Hence, all data with a relative path is submitted into this directory. Therefore to visualize all your tensorboard runs one should run the command: `tensorboard --logdir outputs`. This will then allow you to compare your results across runs.  
You can also [customize](https://hydra.cc/docs/configure_hydra/workdir) your Hydra working directory.

### Multi Run

One nice feature about Hydra in [multi-run](https://hydra.cc/docs/next/tutorials/basic/running_your_app/multi-run/). This can enable you to run your application multiple times with different configurations. A new directory will be created called multirun with the results of these various parameters. You can visualize from tensorboard these results by running: `tensorboard --logdir multirun`.

Other interesting information about Hydra can be found in the [docs](https://hydra.cc/docs/intro/).
