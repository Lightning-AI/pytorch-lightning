#######################################
Eliminate config boilerplate (Advanced)
#######################################


Class type defaults
^^^^^^^^^^^^^^^^^^^

The support for classes as type hints allows to try many possibilities with the same CLI. This is a useful feature, but
it can make it tempting to use an instance of a class as a default. For example:

.. testcode::

    class MyMainModel(LightningModule):
        def __init__(
            self,
            backbone: torch.nn.Module = MyModel(encoder_layers=24),  # BAD PRACTICE!
        ):
            super().__init__()
            self.backbone = backbone

Normally classes are mutable as it is in this case. The instance of :code:`MyModel` would be created the moment that the
module that defines :code:`MyMainModel` is first imported. This means that the default of :code:`backbone` will be
initialized before the CLI class runs :code:`seed_everything` making it non-reproducible. Furthermore, if
:code:`MyMainModel` is used more than once in the same Python process and the :code:`backbone` parameter is not
overridden, the same instance would be used in multiple places which very likely is not what the developer intended.
Having an instance as default also makes it impossible to generate the complete config file since for arbitrary classes
it is not known which arguments were used to instantiate it.

A good solution to these problems is to not have a default or set the default to a special value (e.g. a
string) which would be checked in the init and instantiated accordingly. If a class parameter has no default and the CLI
is subclassed then a default can be set as follows:

.. testcode::

    default_backbone = {
        "class_path": "import.path.of.MyModel",
        "init_args": {
            "encoder_layers": 24,
        },
    }


    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.set_defaults({"model.backbone": default_backbone})

A more compact version that avoids writing a dictionary would be:

.. testcode::

    from jsonargparse import lazy_instance


    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.set_defaults({"model.backbone": lazy_instance(MyModel, encoder_layers=24)})


Argument linking
^^^^^^^^^^^^^^^^

Another case in which it might be desired to extend :class:`~pytorch_lightning.utilities.cli.LightningCLI` is that the
model and data module depend on a common parameter. For example in some cases both classes require to know the
:code:`batch_size`. It is a burden and error prone giving the same value twice in a config file. To avoid this the
parser can be configured so that a value is only given once and then propagated accordingly. With a tool implemented
like shown below, the :code:`batch_size` only has to be provided in the :code:`data` section of the config.

.. testcode::

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("data.batch_size", "model.batch_size")


    cli = MyLightningCLI(MyModel, MyDataModule)

The linking of arguments is observed in the help of the tool, which for this example would look like:

.. code-block:: bash

    $ python trainer.py fit --help
      ...
        --data.batch_size BATCH_SIZE
                              Number of samples in a batch (type: int, default: 8)

      Linked arguments:
        model.batch_size <-- data.batch_size
                              Number of samples in a batch (type: int)

Sometimes a parameter value is only available after class instantiation. An example could be that your model requires
the number of classes to instantiate its fully connected layer (for a classification task) but the value is not
available until the data module has been instantiated. The code below illustrates how to address this.

.. testcode::

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")


    cli = MyLightningCLI(MyClassModel, MyDataModule)

Instantiation links are used to automatically determine the order of instantiation, in this case data first.

.. tip::

    The linking of arguments can be used for more complex cases. For example to derive a value via a function that takes
    multiple settings as input. For more details have a look at the API of `link_arguments
    <https://jsonargparse.readthedocs.io/en/stable/#jsonargparse.core.ArgumentParser.link_arguments>`_.


Variable Interpolation
^^^^^^^^^^^^^^^^^^^^^^

The linking of arguments is intended for things that are meant to be non-configurable. This improves the CLI user
experience since it avoids the need for providing more parameters. A related concept is
variable interpolation which in contrast keeps things being configurable.

The YAML standard defines anchors and aliases which is a way to reuse the content in multiple places of the YAML. This is
supported in the ``LightningCLI`` though it has limitations. Support for OmegaConf's more powerful `variable
interpolation <https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#variable-interpolation>`__ will be available
out of the box if this package is installed. To install it run :code:`pip install omegaconf`. Then to enable the use
of OmegaConf in a ``LightningCLI``, when instantiating a parameter needs to be given for the parser as follows:

.. testcode::

    cli = LightningCLI(MyModel, parser_kwargs={"parser_mode": "omegaconf"})

With the encoder-decoder example model above a possible YAML that uses variable interpolation could be the following:

.. code-block:: yaml

    model:
      encoder_layers: 12
      decoder_layers:
      - ${model.encoder_layers}
      - 4

