# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Any

import torch

from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.utils import to_onehot
from pytorch_lightning.metrics.classification.utils import _input_format_classification, _stat_scores


def _reduce_scores(scores: torch.Tensor, weights: torch.Tensor, average: str):
    """ Reduce scores according to the average method. """

    if average in ["binary", "micro", "none"]:
        return scores
    elif average in ["macro", "samples"]:
        return scores.mean()
    elif average == "weighted":
        w_scores = scores * (weights / weights.sum())
        return w_scores.sum()


class StatScores(Metric):
    """Base metric for (multilabel) classification.

    This metric takes care of computing and updating the number of true
    positives, false positives, true negatives and false negatives according
    to the reduction method. All that the metrics that subclass it need to do
    is to implement the `compute()` method (or set the reduction method in
    init).
    """

    def __init__(
        self,
        average: str = "micro",
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        logits: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.average = average
        self.num_classes = num_classes
        self.threshold = threshold
        self.logits = logits

        if self.num_classes == 1 and self.average != "binary":
            raise ValueError(
                "You can only set number of classes to 1 if average='binary'."\
                "If your data is binary, but you want to treat it as 2 classes, set num_classes to 2."
            )

        if self.average in ["micro", "binary"]:
            default, reduce_fn = torch.tensor(0), "sum"
        elif self.average in ["macro", "weighted", "none"]:
            if not num_classes or num_classes < 1:
                raise ValueError(
                    "When you set the average as macro, weighted or none, you have to provide the number of classes."
                )

            default, reduce_fn = torch.zeros((num_classes,)), "sum"
        elif self.average == "samples":
            default, reduce_fn = torch.empty(0), "cat"
        else:
            raise ValueError(f"Average '{self.average}' is not valid.")

        for s in ("tp", "fp", "tn", "fn"):
            self.add_state(s, default=default.detach().clone(), dist_reduce_fx=reduce_fn)

    def update(self, preds: torch.Tensor, target: torch.Tensor):

        preds, target = _input_format_classification(preds, target, self.threshold, self.num_classes, self.logits)
        if self.average == "binary" and preds.shape[1] > 2:
            raise ValueError(
                f"binary average method is only valid for 2 labels, but preds have {preds.shape[1]} labels."
            )

        if self.average != "binary" and preds.shape[1] == 1:
            preds = to_onehot(preds.view(-1), 2)
            target = to_onehot(target.view(-1), 2)

        tp, fp, tn, fn = _stat_scores(preds, target, average=self.average)

        if self.average in ["binary", "micro", "macro", "weighted", "none"]:
            self.tp += tp
            self.fp += fp
            self.tn += tn
            self.fn += fn

        elif self.average == "samples":
            if isinstance(self.tp, list):
                self.tp, self.fp, self.tn, self.fn = tp, fp, tn, fn
            else:
                print((self.tp, tp))
                self.tp = torch.cat((self.tp, tp))
                self.fp = torch.cat((self.fp, fp))
                self.tn = torch.cat((self.tn, tn))
                self.fn = torch.cat((self.fn, fn))

    def compute(self):
        return self.tp, self.fn, self.tn, self.fn




