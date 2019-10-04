import os
import sys

from test_tube import HyperOptArgumentParser, Experiment
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.arg_parse import add_default_args
from pytorch_lightning.callbacks.pt_callbacks import EarlyStopping, ModelCheckpoint

from examples.new_project_templates.lightning_module_template import LightningTemplateModel


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    :return:
    """
    # init experiment
    exp = Experiment(
        name=hparams.tt_name,
        debug=hparams.debug,
        save_dir=hparams.tt_save_path,
        version=hparams.hpc_exp_number,
        autosave=False,
        description=hparams.tt_description
    )

    exp.argparse(hparams)
    exp.save()

    # build model
    model = LightningTemplateModel(hparams)

    # configure trainer
    trainer = Trainer(experiment=exp)

    # train model
    trainer.fit(model)


if __name__ == '__main__':

    # use default args given by lightning
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = HyperOptArgumentParser(strategy='random_search', add_help=False)
    add_default_args(parent_parser, root_dir)

    # allow model to overwrite or extend args
    parser = LightningTemplateModel.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    # train model
    main(hyperparams)
