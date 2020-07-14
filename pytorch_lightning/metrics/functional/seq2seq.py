import torch
from torchtext.data.metrics import bleu_score
from typing import List

def bleu(
    pred: List[str],
    targ: List[str],
    max_n : int = 4,
    weights : list = [0.25, 0.25, 0.25, 0.25]
) -> torch.Tensor:
    """
    Computes Bleu score.

    Args:
        pred: predicted texts
        target: ground truth texts
        max_n:  the maximum n-gram we want to use. E.g. if max_n=3, we will use unigrams, bigrams and trigrams
        weights: a list of weights used for each n-gram category (uniform by default)
    
    Return:
        Float Tensor with Bleu score.
    """
    return torch.tensor(bleu_score(pred,targ,max_n,weights))
