import argparse

from pl_examples.lite_examples.gan.gan_example import GANTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--precision", type=int, default=32)
    args = parser.parse_args()

    trainer = GANTrainer(**vars(args))
    trainer.run()
