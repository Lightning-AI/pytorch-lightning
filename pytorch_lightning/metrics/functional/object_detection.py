
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
from torch import Tensor

try:
    from torchvision.ops import box_iou
except ModuleNotFoundError:  # pragma: no-cover
    box_iou = None


def mean_average_precision(
    pred_image_indices: Tensor,
    pred_probs: Tensor,
    pred_labels: Tensor,
    pred_bboxes: Tensor,
    target_image_indices: Tensor,
    target_labels: Tensor,
    target_bboxes: Tensor,
    iou_threshold: float,
    ap_calculation: str
) -> Tensor:
    """
    Compute mean average precision for object detection task

    Args:
        pred_image_indices: an (N,)-shaped Tensor of image indices of the predictions
        pred_probs: an (N,)-shaped Tensor of probabilities of the predictions
        pred_labels: an (N,)-shaped Tensor of predicted labels
        pred_bboxes: an (N, 4)-shaped Tensor of predicted bounding boxes
        target_image_indices: an (M,)-shaped Tensor of image indices of the groudn truths
        target_labels: an (M,)-shaped Tensor of ground truth labels
        target_bboxes: an (M, 4)-shaped Tensor of ground truth bounding boxes
        iou_threshold: threshold for IoU score for determining true positive and
                       false positive predictions.
        ap_calculation: method to calculate the average precision of the precision-recall curve

            - ``'step'``: calculate the step function integral, the same way as
            :func:`~pytorch_lightning.metrics.functional.average_precision.average_precision`
            - ``'VOC2007'``: calculate the 11-point sampling of interpolation of the precision recall curve
            - ``'VOC2010'``: calculate the step function integral of the interpolated precision recall curve
            - ``'COCO'``: calculate the 101-point sampling of the interpolated precision recall curve

    Returns:
        mean of the average precision for each class in object detection task.

    """
    if box_iou is None:
        raise ImportError('`mean_average_precision` metric requires `torchvision`, which is not installed. '
                          ' install it with `pip install torchvision`.')
    classes = torch.cat([pred_labels, target_labels]).unique()
    average_precisions = torch.zeros(len(classes))
    for class_idx, c in enumerate(classes):
        desc_indices = torch.argsort(pred_probs, descending=True)[pred_labels == c]
        targets_per_images = Counter([idx.item() for idx in target_image_indices[target_labels == c]])
        targets_assigned = {
            image_idx: torch.zeros(count, dtype=torch.bool) for image_idx, count in targets_per_images.items()
        }
        if len(desc_indices) == 0:
            continue
        tps = torch.zeros(len(desc_indices))
        fps = torch.zeros(len(desc_indices))
        for i, pred_idx in enumerate(desc_indices):
            image_idx = pred_image_indices[pred_idx].item()
            gt_bboxes = target_bboxes[(target_image_indices == image_idx) & (target_labels == c)]
            ious = box_iou(torch.unsqueeze(pred_bboxes[pred_idx], dim=0), gt_bboxes)
            best_iou, best_target_idx = ious.squeeze(0).max(0) if len(gt_bboxes) > 0 else (0, -1)
            if best_iou > iou_threshold and not targets_assigned[image_idx][best_target_idx]:
                targets_assigned[image_idx][best_target_idx] = True
                tps[i] = 1
            else:
                fps[i] = 1
        tps_cum, fps_cum = torch.cumsum(tps, dim=0), torch.cumsum(fps, dim=0)
        precision = tps_cum / (tps_cum + fps_cum)
        num_targets = len(target_labels[target_labels == c])
        recall = tps_cum / num_targets if num_targets else tps_cum
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
        elif ap_calculation == "VOC2010":
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
