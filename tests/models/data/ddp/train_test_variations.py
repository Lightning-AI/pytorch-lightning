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


def variation_fit_fit(trainer, model):
    trainer.fit(model)
    trainer.fit(model)


def variation_test_test(trainer, model):
    trainer.test(model)
    trainer.test(model)


def variation_test_fit_test(trainer, model):
    trainer.test(model)
    trainer.fit(model)
    trainer.test(model)


def get_variations():
    variations = [
        "variation_fit_test",
        "variation_test_fit",
        "variation_fit_fit",
        "variation_test_test",
        "variation_test_fit_test",
    ]
    return variations


def main():
    seed_everything(1234)
    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--variation', default=variation_fit_test.__name__)
    parser.set_defaults(gpus=2)
    parser.set_defaults(distributed_backend="ddp")
    args = parser.parse_args()

    model = EvalModelTemplate()
    trainer = Trainer.from_argparse_args(args)

    # run the chosen variation
    run_variation = globals()[args.variation]
    run_variation(trainer, model)

    # TODO
    # remove this in https://github.com/PyTorchLightning/pytorch-lightning/pull/2165
    # when we have proper signal handling working
    # otherwise we will see zombie processes in CI, causing tests to hang
    for p in trainer.interactive_ddp_procs:
        p.kill()


if __name__ == '__main__':
    main()
