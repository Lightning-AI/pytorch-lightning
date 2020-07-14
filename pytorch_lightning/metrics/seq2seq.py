import torch
from pytorch_lightning.metrics.functional.seq2seq import (
    bleu
)

from pytorch_lightning.metrics.metric import Metric

class Bleu(Metric):
    '''
    Computes the Bleu score.

    Example:
    >>> candidate_corpus = [['My', 'full', 'pl', 'test'], ['Another', 'Sentence']]
    >>> references_corpus = [[['My', 'full', 'pl', 'test'], ['Completely', 'Different']], [['No', 'Match']]]
    >>> metric = Bleu()
    >>> metric(candidate_corpus, references_corpus)
    tensor(0.8409)
    
    '''

    def __init__(self,max_n : int = 4,weights : list = [0.25, 0.25, 0.25, 0.25]):
        super().__init__(name='bleu')
        self.max_n=max_n 
        self.weights = weights

    def forward(self,x,y):
        return bleu(x,y,self.max_n,self.weights)
