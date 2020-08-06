"""
Runs several combinations of `.fit()` and `.test()` on a single node across multiple gpus.
"""
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from tests.base import EvalModelTemplate


def variation_fit_test(trainer, model):
    trainer.fit(model)
    trainer.test(model)


def variation_test_fit(trainer, model):
    trainer.test(model)
    trainer.fit(model)


def variation_test_test(trainer, model):
    trainer.test(model)
    trainer.test(model)


def variation_test_fit_test(trainer, model):
    trainer.test(model)
    trainer.fit(model)
    trainer.test(model)


def get_variations():
    variations = [v for v in locals() if v.startswith("variation")]
    return variations


def main():
    seed_everything(1234)
    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('variation', default=variation_fit_test.__name__, required=True)
    parser.set_defaults(gpus=2)
    parser.set_defaults(distributed_backend="ddp")
    args = parser.parse_args()

    model = EvalModelTemplate()
    trainer = Trainer.from_argparse_args(args)

    # run the chosen variation
    run_variation = locals()[args.variation]
    run_variation(trainer, model)


if __name__ == '__main__':
    main()
