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
from pytorch_lightning.metrics.functional.classification import (
    accuracy,
    auc,
    auroc,
    average_precision,
    confusion_matrix,
    dice_score,
    f1_score,
    fbeta_score,
    multiclass_precision_recall_curve,
    multiclass_roc,
    precision,
    precision_recall,
    precision_recall_curve,
    recall,
    roc,
    stat_scores,
    stat_scores_multiple_classes,
    to_categorical,
    to_onehot,
    iou,
)
from pytorch_lightning.metrics.functional.nlp import bleu_score
from pytorch_lightning.metrics.functional.regression import (
    mae,
    mse,
    psnr,
    rmse,
    rmsle,
    ssim
)
from pytorch_lightning.metrics.functional.self_supervised import (
    embedding_similarity
)
