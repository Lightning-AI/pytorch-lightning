import pytest
import torch

from pytorch_lightning.metrics.functional.object_detection import mean_average_precision


@pytest.mark.parametrize(
    [
        "pred_image_indices",
        "pred_probs",
        "pred_labels",
        "pred_bboxes",
        "target_image_indices",
        "target_labels",
        "target_bboxes",
        "iou_threshold",
    ],
    [
        pytest.param(
            torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.int),
            torch.tensor([0.6, 0.1, 0.9, 0.2, 0.65, 0.7], dtype=torch.float),
            torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.int),
            torch.tensor([
                [62, 83, 225, 195],
                [79, 93, 118, 131],
                [117, 192, 127, 244],
                [15, 201, 26, 254],
                [35, 45, 210, 170],
                [210, 80, 295, 90]
            ], dtype=torch.float),
            torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.int),
            torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.int),
            torch.tensor([
                [100, 50, 205, 200],
                [85, 83, 225, 195],
                [117, 192, 127, 244],
                [10, 200, 15, 254],
                [30, 40, 234, 150],
                [210, 80, 295, 90],
            ], dtype=torch.float),
            0.5,
        )
    ]
)
@pytest.mark.parametrize(
    "ap_calculation, expected_map",
    [
        pytest.param("step", torch.tensor([2 / 3])),
        pytest.param("VOC2007", torch.tensor([7 / 11])),
        pytest.param("VOC2010", torch.tensor([2 / 3])),
        pytest.param("COCO", torch.tensor([67 / 101]))
    ]
)
def test_mean_average_precision_0(
    pred_image_indices,
    pred_probs,
    pred_labels,
    pred_bboxes,
    target_image_indices,
    target_labels,
    target_bboxes,
    iou_threshold,
    ap_calculation,
    expected_map
):
    mAP = mean_average_precision(pred_image_indices, pred_probs, pred_labels, pred_bboxes, target_image_indices, target_labels, target_bboxes, iou_threshold, ap_calculation)
    assert torch.allclose(mAP, expected_map)


@pytest.mark.parametrize(
    [
        "pred_image_indices",
        "pred_probs",
        "pred_labels",
        "pred_bboxes",
        "target_image_indices",
        "target_labels",
        "target_bboxes",
        "iou_threshold",
    ],
    [
        pytest.param(
            torch.tensor([0, 1, 0, 2], dtype=torch.int),
            torch.tensor([0.9, 0.9, 0.9, 0.9], dtype=torch.float),
            torch.tensor([1, 0, 2, 1], dtype=torch.int),
            torch.tensor([
                [100, 100, 200, 200],
                [100, 100, 200, 200],
                [100, 100, 200, 200],
                [100, 100, 200, 200],
            ], dtype=torch.float),
            torch.tensor([0, 1, 0, 2], dtype=torch.int),
            torch.tensor([1, 0, 2, 1], dtype=torch.int),
            torch.tensor([
                [100, 100, 200, 200],
                [100, 100, 200, 200],
                [100, 100, 200, 200],
                [100, 100, 200, 200],
            ], dtype=torch.float),
            0.5,
        )
    ]
)
@pytest.mark.parametrize(
    "ap_calculation, expected_map",
    [
        pytest.param("step", torch.tensor([1.])),
        pytest.param("VOC2007", torch.tensor([1.])),
        pytest.param("VOC2010", torch.tensor([1.])),
        pytest.param("COCO", torch.tensor([1.]))
    ]
)
def test_mean_average_precision_1(
    pred_image_indices,
    pred_probs,
    pred_labels,
    pred_bboxes,
    target_image_indices,
    target_labels,
    target_bboxes,
    iou_threshold,
    ap_calculation,
    expected_map
):
    mAP = mean_average_precision(pred_image_indices, pred_probs, pred_labels, pred_bboxes, target_image_indices, target_labels, target_bboxes, iou_threshold, ap_calculation)
    assert torch.allclose(mAP, expected_map)


@pytest.mark.parametrize(
    [
        "pred_image_indices",
        "pred_probs",
        "pred_labels",
        "pred_bboxes",
        "target_image_indices",
        "target_labels",
        "target_bboxes",
        "iou_threshold",
    ],
    [
        pytest.param(
            torch.tensor([0, 1, 0, 2], dtype=torch.int),
            torch.tensor([0.9, 0.9, 0.9, 0.9], dtype=torch.float),
            torch.tensor([1, 0, 2, 1], dtype=torch.int),
            torch.tensor([
                [100, 100, 200, 200],
                [100, 100, 200, 200],
                [100, 100, 200, 200],
                [100, 100, 200, 200],
            ], dtype=torch.float),
            torch.tensor([0], dtype=torch.int),
            torch.tensor([3], dtype=torch.int),
            torch.tensor([[100, 100, 200, 200]], dtype=torch.float),
            0.5
        )
    ]
)
@pytest.mark.parametrize(
    "ap_calculation, expected_map",
    [
        pytest.param("step", torch.tensor([0.])),
        pytest.param("VOC2007", torch.tensor([0.])),
        pytest.param("VOC2010", torch.tensor([0.])),
        pytest.param("COCO", torch.tensor([0.]))
    ]
)
def test_mean_average_precision_no_target(
    pred_image_indices,
    pred_probs,
    pred_labels,
    pred_bboxes,
    target_image_indices,
    target_labels,
    target_bboxes,
    iou_threshold,
    ap_calculation,
    expected_map
):
    mAP = mean_average_precision(pred_image_indices, pred_probs, pred_labels, pred_bboxes, target_image_indices, target_labels, target_bboxes, iou_threshold, ap_calculation)
    assert torch.allclose(mAP, expected_map)
