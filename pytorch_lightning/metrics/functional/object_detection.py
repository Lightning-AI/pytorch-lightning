
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
from collections import Counter

import torch

try:
    from torchvision.ops import box_iou
except ModuleNotFoundError:  #pragma: no-cover
    box_iou = None


def mean_average_precision(
        preds: torch.Tensor, target: torch.Tensor, iou_threshold: float, ap_calculation: str
) -> torch.Tensor:
    """
    Compute mean average precision for object detection task

    Args:
        preds: an Nx7 batch of predictions with representation
               ``[image_idx, class_pred, class_prob, x_min, y_min, x_max, y_max]``
        target: an Nx6 batch of targets with representation
                ``[image_idx, class_label, x_min, y_min, x_max, y_max]``
        iou_threshold: threshold for IoU score for determining true positive and
                       false positive predictions.
        ap_calculation: one of "step", "VOC2007", "VOC2012", or "COCO"

    Returns:
        mean of the average precision for each class in object detection task.
    """
    if box_iou is None:
        raise ImportError('You want to use `torchvision` which is not installed yet,'
                          ' install it with `pip install torchvision`.')
    classes = torch.cat([preds[:, 1], target[:, 1]]).unique()
    average_precisions = torch.zeros(len(classes))
    for class_idx, c in enumerate(classes):
        c_preds = sorted(preds[preds[:, 1] == c], key=lambda x: x[2], reverse=True)
        c_target = target[target[:, 1] == c]
        targets_per_images = Counter([t[0].item() for t in c_target])
        targets_assigned = {
            image_idx: torch.zeros(count, dtype=torch.bool) for image_idx, count in targets_per_images.items()
        }
        tps = torch.zeros(len(c_preds))
        fps = torch.zeros(len(c_preds))
        if len(c_preds) == 0:
            continue
        for i, p in enumerate(c_preds):
            image_idx = p[0].item()
            ground_truths = c_target[c_target[:, 0] == image_idx]
            ious = box_iou(p[None, 3:], ground_truths[:, 2:])
            best_iou, best_target_idx = ious.squeeze(0).max(0) if len(ground_truths) > 0 else (0, -1)
            if best_iou > iou_threshold and not targets_assigned[image_idx][best_target_idx]:
                targets_assigned[image_idx][best_target_idx] = True
                tps[i] = 1
            else:
                fps[i] = 1
        tps_cum, fps_cum = torch.cumsum(tps, dim=0), torch.cumsum(fps, dim=0)
        precision = tps_cum / (tps_cum + fps_cum)
        recall = tps_cum / len(c_target) if len(c_target) else tps_cum
        precision = torch.cat([reversed(precision), torch.tensor([1.])])
        recall = torch.cat([reversed(recall), torch.tensor([0.])])
        if ap_calculation == "step":
            average_precision = -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])
        elif ap_calculation == "VOC2007":
            average_precision = 0
            recall_thresholds = torch.linspace(0, 1, 11)
            for threshold in recall_thresholds:
                points = precision[:-1][recall[:-1] >= threshold]
                average_precision += torch.max(points) / 11 if len(points) else 0
        elif ap_calculation == "VOC2012":
            average_precision = 0
            for i in range(len(precision)):
                precision[i] = torch.max(precision[:i + 1])
            average_precision = -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])
        elif ap_calculation == "COCO":
            average_precision = 0
            recall_thresholds = torch.linspace(0, 1, 101)
            for threshold in recall_thresholds:
                points = precision[:-1][recall[:-1] >= threshold]
                average_precision += torch.max(points) / 101 if len(points) else 0
        else:
            raise NotImplementedError(f"'{ap_calculation}' is not supported.")
        average_precisions[class_idx] = average_precision
    mean_average_precision = torch.mean(average_precisions)
    return mean_average_precision
