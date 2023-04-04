import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.demos import DemoLanguageModel, WikiText2


def main():
    L.seed_everything(42)
    train_dataset = WikiText2("data/wikitext-2/train.txt")
    train_dataloader = DataLoader(train_dataset, batch_size=20)
    model = DemoLanguageModel(ntokens=len(train_dataset.dictionary))
    trainer = L.Trainer(gradient_clip_val=0.25, max_epochs=20)
    trainer.fit(model, train_dataloader)


if __name__ == "__main__":
    main()
