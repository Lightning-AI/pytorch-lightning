import argparse
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from shutil import copyfile

keys_mapping = {
    "checkpoint_callback_best_model_score": (type(ModelCheckpoint), "best_model_score"),
    "checkpoint_callback_best_model_path": (type(ModelCheckpoint), "best_model_path"),
    "checkpoint_callback_best": (type(ModelCheckpoint), "wait_count"),
    "early_stop_callback_wait": (type(EarlyStopping), "best_model_score"),
    "early_stop_callback_patience": (type(EarlyStopping), "patience"),
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Upgrade an old checkpoint to the current schema.")
    parser.add_argument("--file", help="filepath for a checkpoint to upgrade")
    args = parser.parse_args()

    checkpoint = torch.load(args.file)
    copyfile(args.file, args.file + ".bak")

    for key, new_path in keys_mapping.items():
        value = checkpoint[key]
        del checkpoint[key]
        callback_type, callback_key = new_path
        checkpoint["callbacks"][callback_type][callback_key] = value

    torch.save(checkpoint, args.file)
