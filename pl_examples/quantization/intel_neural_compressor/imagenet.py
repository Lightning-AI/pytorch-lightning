# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This example is largely adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py.

Before you can run this example, you will need to download the ImageNet dataset manually from the
`official website <http://image-net.org/download>`_ and place it into a folder `path/to/imagenet`.

Train on ImageNet with default parameters:

.. code-block: bash

    python imagenet.py --data-path /path/to/imagenet

or show all options you can change:

.. code-block: bash

    python imagenet.py --help
"""
import pytorch_lightning as pl
from pl_examples import cli_lightning_logo
from pl_examples.domain_templates.imagenet import ImageNetLightningModel
from pytorch_lightning.utilities.cli import LightningCLI


def run_cli():
    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_argument(
                "-e", "--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set"
            )

    cli = MyLightningCLI(ImageNetLightningModel, seed_everything_default=42, save_config_overwrite=True, run=False)
    if cli.config["evaluate"]:
        from neural_compressor.utils.pytorch import load

        print("Load quantized configure from ", cli.trainer.default_root_dir)
        cli.model.model = load(cli.trainer.default_root_dir, cli.model.model)
        out = cli.trainer.validate(cli.model, datamodule=cli.datamodule)
        print("val_acc1:{}".format(out[0]["val_acc1"]))
    else:
        callback = pl.callbacks.INCQuantization(
            "config/quantization.yaml",
            monitor="val_acc1",
            module_name_to_quant="model",
            dirpath=cli.trainer.default_root_dir,
        )
        cli.trainer.callbacks.append(callback)
        cli.trainer.fit(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_lightning_logo()
    run_cli()
