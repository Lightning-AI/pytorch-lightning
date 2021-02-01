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
from pytorch_lightning.metrics.classification import (  # noqa: F401
    Accuracy,
    AUC,
    AUROC,
    AveragePrecision,
    ConfusionMatrix,
    F1,
    FBeta,
    HammingDistance,
    IoU,
    Precision,
    PrecisionRecallCurve,
    Recall,
    ROC,
    StatScores,
)
from pytorch_lightning.metrics.metric import Metric, MetricCollection  # noqa: F401
from pytorch_lightning.metrics.regression import (  # noqa: F401
    ExplainedVariance,
    MeanAbsoluteError,
    MeanSquaredError,
    MeanSquaredLogError,
    PSNR,
    R2Score,
    SSIM,
)
