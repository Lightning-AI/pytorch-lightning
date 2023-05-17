from typing import Any

import datasets
import torch
import transformers

import lightning.pytorch as pl


class Collate:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        inputs = [example["question"] for example in examples]
        encodings = self.tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return dict(encodings)


class DummyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.bert = None

    def configure_sharded_model(self):
        self.bert = transformers.AutoModel.from_pretrained("bert-base-uncased")

    def forward(self, batch: dict[str, torch.Tensor]):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        output = self.bert(input_ids, attention_mask=attention_mask)
        loss = output.pooler_output.mean()
        return loss

    def training_step(self, batch: dict[str, Any], *args: Any, **kwargs: Any):
        loss = self.forward(batch)
        return loss

    def validation_step(self, batch: dict[str, Any], *args: Any, **kwargs: Any):
        loss = self.forward(batch)
        return loss

    def test_step(self, batch: dict[str, Any], *args: Any, **kwargs: Any):
        loss = self.forward(batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)
        # return deepspeed.ops.adam.DeepSpeedCPUAdam(self.parameters(), lr=1e-5)


if __name__ == "__main__":
    dataset = datasets.DatasetDict(
        {
            "train": datasets.load_dataset("squad", split="train[:1%]"),
            "validation": datasets.load_dataset("squad", split="validation[:1%]"),
        }
    )
    model = DummyModel()
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=32,
        collate_fn=Collate(tokenizer),
        num_workers=4,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset["validation"],
        batch_size=32,
        collate_fn=Collate(tokenizer),
        num_workers=4,
    )

    trainer = pl.Trainer(
        devices=2,
        accelerator="gpu",
        precision="16-mixed",
        strategy=pl.strategies.DeepSpeedStrategy(stage=3),
    )

    trainer.test(
        model, dataloaders=val_dataloader
    )  # <-- no problem when commenting this, the problem is probably related to `lightning`
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
