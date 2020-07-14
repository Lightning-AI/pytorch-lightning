import torch
from pytorch_lightning.metrics.seq2seq import Bleu

def test_bleu():
    candidate_corpus = [['My', 'full', 'pl', 'test'], ['Another', 'Sentence']]
    references_corpus = [[['My', 'full', 'pl', 'test'], ['Completely', 'Different']], [['No', 'Match']]]
    metric = Bleu()
    score = metric(candidate_corpus, references_corpus)
    assert isinstance(score, torch.Tensor)
