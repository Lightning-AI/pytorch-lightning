import os

import torch
from espnet.bin.asr_train import get_parser
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from lhotse import CutSet, Fbank, FbankConfig
from lhotse.dataset import BucketingSampler, OnTheFlyFeatures
from lhotse.dataset.collation import TokenCollater
from lhotse.recipes import download_librispeech, prepare_librispeech
from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule, LightningModule, seed_everything, Trainer

parser = get_parser()
parser = E2E.add_arguments(parser)
args = parser.parse_args(
    ["--mtlalpha", "0.0", "--outdir", "out", "--dict", ""]  # weight for cross entropy and CTC loss
)  # TODO: allow no arg


class MinimalEspnetDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: TokenCollater):
        self.extractor = OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
        self.tokenizer = tokenizer

    def __getitem__(self, cuts: CutSet) -> dict:
        cuts = cuts.sort_by_duration()
        feats, feat_lens = self.extractor(cuts)
        tokens, token_lens = self.tokenizer(cuts)
        return {"xs_pad": feats, "ilens": feat_lens, "ys_pad": tokens}


class DataModule(LightningDataModule):
    def prepare_data(
        self,
    ) -> None:
        download_librispeech(dataset_parts="mini_librispeech")

    def setup(self, stage=None):
        libri = prepare_librispeech(corpus_dir="LibriSpeech", output_dir="data/")
        self.cuts_train = CutSet.from_manifests(**libri["train-clean-5"])
        self.cuts_test = CutSet.from_manifests(**libri["dev-clean-2"])
        self.tokenizer = TokenCollater(self.cuts_train)
        self.tokenizer(self.cuts_test.subset(first=2))
        self.tokenizer.inverse(*self.tokenizer(self.cuts_test.subset(first=2)))

    def train_dataloader(self):
        train_sampler = BucketingSampler(
            self.cuts_train, max_duration=300, shuffle=True, bucket_method="equal_duration"
        )
        return DataLoader(MinimalEspnetDataset(self.tokenizer), sampler=train_sampler, batch_size=None, num_workers=1)

    def test_dataloader(self):
        test_sampler = BucketingSampler(self.cuts_test, max_duration=400, shuffle=False, bucket_method="equal_duration")
        return DataLoader(MinimalEspnetDataset(self.tokenizer), sampler=test_sampler, batch_size=None, num_workers=1)

    @property
    def model_kwargs(self):
        return {
            "odim": len(self.tokenizer.idx2token),
        }


trainer = Trainer(strategy=None, gpus=1, logger=None, replace_sampler_ddp=False)
os.environ["PL_FAULT_TOLERANT_TRAINING"] = "2"

seed_everything(42)


class Model(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None) -> None:
        datamodule = self.trainer.datamodule
        setattr(self.args, "char_list", list(datamodule.tokenizer.idx2token))
        self.model = E2E(80, **datamodule.model_kwargs, args=self.args)

    def training_step(self, batch, batch_idx):
        return self.model(**batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())


dm = DataModule()
model = Model(args)

trainer.fit(model, dm)
