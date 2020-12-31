import pytest
import torch

from pytorch_lightning.metrics.functional.object_detection import mean_average_precision


@pytest.mark.parametrize(
    ["pred", "target", "iou_threshold"],
    [
        pytest.param(
            torch.tensor([
                [0, 0, 0.6, 62, 83, 225, 195],
                [0, 0, 0.1, 79, 93, 118, 131],
                [0, 0, 0.9, 117, 192, 127, 244],
                [0, 0, 0.2, 15, 201, 26, 254],
                [0, 0, 0.65, 35, 45, 210, 170],
                [0, 0, 0.7, 210, 80, 295, 90]
            ], dtype=torch.float),
            torch.tensor([
                [0, 0, 100, 50, 205, 200],
                [0, 0, 85, 83, 225, 195],
                [0, 0, 117, 192, 127, 244],
                [0, 0, 10, 200, 15, 254],
                [0, 0, 30, 40, 234, 150],
                [0, 0, 210, 80, 295, 90],
            ], dtype=torch.float),
            0.5,
        ),
    ]
)
@pytest.mark.parametrize(
    "ap_calculation, expected_map",
    [
        pytest.param("step", torch.tensor([2 / 3])),
        pytest.param("VOC2007", torch.tensor([7 / 11])),
        pytest.param("VOC2012", torch.tensor([2 / 3])),
        pytest.param("COCO", torch.tensor([67 / 101]))
    ]
)
def test_mean_average_precision_0(pred, target, iou_threshold, ap_calculation, expected_map):
    mAP = mean_average_precision(pred, target, iou_threshold, ap_calculation)
    assert torch.allclose(mAP, expected_map)


@pytest.mark.parametrize(
    ["pred", "target", "iou_threshold"],
    [
        pytest.param(
            torch.tensor([
                [0, 1, 0.9, 100, 100, 200, 200],
                [1, 0, 0.9, 100, 100, 200, 200],
                [0, 2, 0.9, 100, 100, 200, 200],
                [2, 1, 0.9, 100, 100, 200, 200],
            ], dtype=torch.float),
            torch.tensor([
                [0, 1, 100., 100., 200., 200.],
                [1, 0, 100., 100., 200., 200.],
                [0, 2, 100., 100., 200., 200.],
                [2, 1, 100., 100., 200., 200.],
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
        pytest.param("VOC2012", torch.tensor([1.])),
        pytest.param("COCO", torch.tensor([1.]))
    ]
)
def test_mean_average_precision_1(pred, target, iou_threshold, ap_calculation, expected_map):
    mAP = mean_average_precision(pred, target, iou_threshold, ap_calculation)
    assert torch.allclose(mAP, expected_map)


@pytest.mark.parametrize(
    ["pred", "target", "iou_threshold"],
    [
        pytest.param(
            torch.tensor([
                [0, 1, 0.9, 100, 100, 200, 200],
                [1, 0, 0.9, 100, 100, 200, 200],
                [0, 2, 0.9, 100, 100, 200, 200],
                [2, 1, 0.9, 100, 100, 200, 200],
            ], dtype=torch.float),
            torch.tensor([[0, 3, 100, 100, 200, 200]], dtype=torch.float),
            0.5,
        )
    ]
)
@pytest.mark.parametrize(
    "ap_calculation, expected_map",
    [
        pytest.param("step", torch.tensor([0.])),
        pytest.param("VOC2007", torch.tensor([0.])),
        pytest.param("VOC2012", torch.tensor([0.])),
        pytest.param("COCO", torch.tensor([0.]))
    ]
)
def test_mean_average_precision_no_target(pred, target, iou_threshold, ap_calculation, expected_map):
    mAP = mean_average_precision(pred, target, iou_threshold, ap_calculation)
    assert torch.allclose(mAP, expected_map)


