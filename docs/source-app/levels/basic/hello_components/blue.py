# app.py
# ! pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/master.zip numpy pandas sentencepiece 'torch>=1.7.0,!=1.8.0' scikit-learn 'transformers==4.16.2'
# ! curl https://gist.githubusercontent.com/awaelchli/49ad15ddebb3449f859a184bd90cea6f/raw/9b4257c938ddd35ddf26009c1c587876445a4c4c/model.py -o model.py
import lightning as L
from lightning.app.components import LightningTrainerMultiNode
from transformers import T5ForConditionalGeneration
from transformers import T5TokenizerFast as T5Tokenizer
from model import TextSummarization, TextSummarizationDataModule, predict

sample_text = """
copy paste your text here
"""


class TLDR(L.LightningWork):
    def run(self):
        t5 = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")

        datamodule = TextSummarizationDataModule(t5_tokenizer)
        model = TextSummarization(model=t5, tokenizer=t5_tokenizer)
        trainer = L.Trainer(max_steps=5)
        trainer.fit(model, datamodule)

        if trainer.global_rank == 0:
            predictions = predict(model.to("cuda"), sample_text)
            print("predictions:", predictions[0])


app = L.LightningApp(
    LightningTrainerMultiNode(
        TLDR,
        num_nodes=2,
        cloud_compute=L.CloudCompute("gpu"),  # gpu-fast-multi
    )
)
