from distutils.version import LooseVersion
from operator import neg, pos

import pytest
import torch

from pytorch_lightning.metrics.compositional import CompositionalMetric
from pytorch_lightning.metrics.metric import Metric

_MARK_TORCH_LOWER_1_4 = dict(
    condition=LooseVersion(torch.__version__) < LooseVersion("1.5.0"), reason='required PT >= 1.5'
)
_MARK_TORCH_LOWER_1_5 = dict(
    condition=LooseVersion(torch.__version__) < LooseVersion("1.6.0"), reason='required PT >= 1.6'
)


class DummyMetric(Metric):

    def __init__(self, val_to_return):
        super().__init__()
        self._num_updates = 0
        self._val_to_return = val_to_return

    def update(self, *args, **kwargs) -> None:
        self._num_updates += 1

    def compute(self):
        return torch.tensor(self._val_to_return)

    def reset(self):
        self._num_updates = 0
        return super().reset()


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), torch.tensor(4)),
        (2, torch.tensor(4)),
        (2.0, torch.tensor(4.0)),
        (torch.tensor(2), torch.tensor(4)),
    ],
)
def test_metrics_add(second_operand, expected_result):
    first_metric = DummyMetric(2)

    final_add = first_metric + second_operand
    final_radd = second_operand + first_metric

    assert isinstance(final_add, CompositionalMetric)
    assert isinstance(final_radd, CompositionalMetric)

    assert torch.allclose(expected_result, final_add.compute())
    assert torch.allclose(expected_result, final_radd.compute())


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [(DummyMetric(3), torch.tensor(2)), (3, torch.tensor(2)), (3, torch.tensor(2)), (torch.tensor(3), torch.tensor(2))],
)
@pytest.mark.skipif(**_MARK_TORCH_LOWER_1_4)
def test_metrics_and(second_operand, expected_result):
    first_metric = DummyMetric(2)

    final_and = first_metric & second_operand
    final_rand = second_operand & first_metric

    assert isinstance(final_and, CompositionalMetric)
    assert isinstance(final_rand, CompositionalMetric)

    assert torch.allclose(expected_result, final_and.compute())
    assert torch.allclose(expected_result, final_rand.compute())


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), torch.tensor(True)),
        (2, torch.tensor(True)),
        (2.0, torch.tensor(True)),
        (torch.tensor(2), torch.tensor(True)),
    ],
)
def test_metrics_eq(second_operand, expected_result):
    first_metric = DummyMetric(2)

    final_eq = first_metric == second_operand

    assert isinstance(final_eq, CompositionalMetric)

    # can't use allclose for bool tensors
    assert (expected_result == final_eq.compute()).all()


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), torch.tensor(2)),
        (2, torch.tensor(2)),
        (2.0, torch.tensor(2.0)),
        (torch.tensor(2), torch.tensor(2)),
    ],
)
@pytest.mark.skipif(**_MARK_TORCH_LOWER_1_4)
def test_metrics_floordiv(second_operand, expected_result):
    first_metric = DummyMetric(5)

    final_floordiv = first_metric // second_operand

    assert isinstance(final_floordiv, CompositionalMetric)

    assert torch.allclose(expected_result, final_floordiv.compute())


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), torch.tensor(True)),
        (2, torch.tensor(True)),
        (2.0, torch.tensor(True)),
        (torch.tensor(2), torch.tensor(True)),
    ],
)
def test_metrics_ge(second_operand, expected_result):
    first_metric = DummyMetric(5)

    final_ge = first_metric >= second_operand

    assert isinstance(final_ge, CompositionalMetric)

    # can't use allclose for bool tensors
    assert (expected_result == final_ge.compute()).all()


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), torch.tensor(True)),
        (2, torch.tensor(True)),
        (2.0, torch.tensor(True)),
        (torch.tensor(2), torch.tensor(True)),
    ],
)
def test_metrics_gt(second_operand, expected_result):
    first_metric = DummyMetric(5)

    final_gt = first_metric > second_operand

    assert isinstance(final_gt, CompositionalMetric)

    # can't use allclose for bool tensors
    assert (expected_result == final_gt.compute()).all()


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), torch.tensor(False)),
        (2, torch.tensor(False)),
        (2.0, torch.tensor(False)),
        (torch.tensor(2), torch.tensor(False)),
    ],
)
def test_metrics_le(second_operand, expected_result):
    first_metric = DummyMetric(5)

    final_le = first_metric <= second_operand

    assert isinstance(final_le, CompositionalMetric)

    # can't use allclose for bool tensors
    assert (expected_result == final_le.compute()).all()


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), torch.tensor(False)),
        (2, torch.tensor(False)),
        (2.0, torch.tensor(False)),
        (torch.tensor(2), torch.tensor(False)),
    ],
)
def test_metrics_lt(second_operand, expected_result):
    first_metric = DummyMetric(5)

    final_lt = first_metric < second_operand

    assert isinstance(final_lt, CompositionalMetric)

    # can't use allclose for bool tensors
    assert (expected_result == final_lt.compute()).all()


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [(DummyMetric([2, 2, 2]), torch.tensor(12)), (torch.tensor([2, 2, 2]), torch.tensor(12))],
)
def test_metrics_matmul(second_operand, expected_result):
    first_metric = DummyMetric([2, 2, 2])

    final_matmul = first_metric @ second_operand

    assert isinstance(final_matmul, CompositionalMetric)

    assert torch.allclose(expected_result, final_matmul.compute())


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), torch.tensor(1)),
        (2, torch.tensor(1)),
        (2.0, torch.tensor(1)),
        (torch.tensor(2), torch.tensor(1)),
    ],
)
def test_metrics_mod(second_operand, expected_result):
    first_metric = DummyMetric(5)

    final_mod = first_metric % second_operand

    assert isinstance(final_mod, CompositionalMetric)
    # prevent Runtime error for PT 1.8 - Long did not match Float
    assert torch.allclose(expected_result.to(float), final_mod.compute().to(float))


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), torch.tensor(4)),
        (2, torch.tensor(4)),
        (2.0, torch.tensor(4.0)),
        (torch.tensor(2), torch.tensor(4)),
    ],
)
def test_metrics_mul(second_operand, expected_result):
    first_metric = DummyMetric(2)

    final_mul = first_metric * second_operand
    final_rmul = second_operand * first_metric

    assert isinstance(final_mul, CompositionalMetric)
    assert isinstance(final_rmul, CompositionalMetric)

    assert torch.allclose(expected_result, final_mul.compute())
    assert torch.allclose(expected_result, final_rmul.compute())


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), torch.tensor(False)),
        (2, torch.tensor(False)),
        (2.0, torch.tensor(False)),
        (torch.tensor(2), torch.tensor(False)),
    ],
)
def test_metrics_ne(second_operand, expected_result):
    first_metric = DummyMetric(2)

    final_ne = first_metric != second_operand

    assert isinstance(final_ne, CompositionalMetric)

    # can't use allclose for bool tensors
    assert (expected_result == final_ne.compute()).all()


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [(DummyMetric([1, 0, 3]), torch.tensor([-1, -2, 3])), (torch.tensor([1, 0, 3]), torch.tensor([-1, -2, 3]))],
)
@pytest.mark.skipif(**_MARK_TORCH_LOWER_1_4)
def test_metrics_or(second_operand, expected_result):
    first_metric = DummyMetric([-1, -2, 3])

    final_or = first_metric | second_operand
    final_ror = second_operand | first_metric

    assert isinstance(final_or, CompositionalMetric)
    assert isinstance(final_ror, CompositionalMetric)

    assert torch.allclose(expected_result, final_or.compute())
    assert torch.allclose(expected_result, final_ror.compute())


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        pytest.param(DummyMetric(2), torch.tensor(4)),
        pytest.param(2, torch.tensor(4)),
        pytest.param(2.0, torch.tensor(4.0), marks=pytest.mark.skipif(**_MARK_TORCH_LOWER_1_5)),
        pytest.param(torch.tensor(2), torch.tensor(4)),
    ],
)
def test_metrics_pow(second_operand, expected_result):
    first_metric = DummyMetric(2)

    final_pow = first_metric**second_operand

    assert isinstance(final_pow, CompositionalMetric)

    assert torch.allclose(expected_result, final_pow.compute())


@pytest.mark.parametrize(
    ["first_operand", "expected_result"],
    [(5, torch.tensor(2)), (5.0, torch.tensor(2.0)), (torch.tensor(5), torch.tensor(2))],
)
@pytest.mark.skipif(**_MARK_TORCH_LOWER_1_4)
def test_metrics_rfloordiv(first_operand, expected_result):
    second_operand = DummyMetric(2)

    final_rfloordiv = first_operand // second_operand

    assert isinstance(final_rfloordiv, CompositionalMetric)
    assert torch.allclose(expected_result, final_rfloordiv.compute())


@pytest.mark.parametrize(["first_operand", "expected_result"], [(torch.tensor([2, 2, 2]), torch.tensor(12))])
def test_metrics_rmatmul(first_operand, expected_result):
    second_operand = DummyMetric([2, 2, 2])

    final_rmatmul = first_operand @ second_operand

    assert isinstance(final_rmatmul, CompositionalMetric)

    assert torch.allclose(expected_result, final_rmatmul.compute())


@pytest.mark.parametrize(["first_operand", "expected_result"], [(torch.tensor(2), torch.tensor(2))])
def test_metrics_rmod(first_operand, expected_result):
    second_operand = DummyMetric(5)

    final_rmod = first_operand % second_operand

    assert isinstance(final_rmod, CompositionalMetric)

    assert torch.allclose(expected_result, final_rmod.compute())


@pytest.mark.parametrize(
    "first_operand,expected_result",
    [
        pytest.param(DummyMetric(2), torch.tensor(4)),
        pytest.param(2, torch.tensor(4)),
        pytest.param(2.0, torch.tensor(4.0), marks=pytest.mark.skipif(**_MARK_TORCH_LOWER_1_5)),
    ],
)
def test_metrics_rpow(first_operand, expected_result):
    second_operand = DummyMetric(2)

    final_rpow = first_operand**second_operand

    assert isinstance(final_rpow, CompositionalMetric)

    assert torch.allclose(expected_result, final_rpow.compute())


@pytest.mark.parametrize(
    ["first_operand", "expected_result"],
    [
        (DummyMetric(3), torch.tensor(1)),
        (3, torch.tensor(1)),
        (3.0, torch.tensor(1.0)),
        (torch.tensor(3), torch.tensor(1)),
    ],
)
def test_metrics_rsub(first_operand, expected_result):
    second_operand = DummyMetric(2)

    final_rsub = first_operand - second_operand

    assert isinstance(final_rsub, CompositionalMetric)

    assert torch.allclose(expected_result, final_rsub.compute())


@pytest.mark.parametrize(
    ["first_operand", "expected_result"],
    [
        (DummyMetric(6), torch.tensor(2.0)),
        (6, torch.tensor(2.0)),
        (6.0, torch.tensor(2.0)),
        (torch.tensor(6), torch.tensor(2.0)),
    ],
)
@pytest.mark.skipif(**_MARK_TORCH_LOWER_1_4)
def test_metrics_rtruediv(first_operand, expected_result):
    second_operand = DummyMetric(3)

    final_rtruediv = first_operand / second_operand

    assert isinstance(final_rtruediv, CompositionalMetric)

    assert torch.allclose(expected_result, final_rtruediv.compute())


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), torch.tensor(1)),
        (2, torch.tensor(1)),
        (2.0, torch.tensor(1.0)),
        (torch.tensor(2), torch.tensor(1)),
    ],
)
def test_metrics_sub(second_operand, expected_result):
    first_metric = DummyMetric(3)

    final_sub = first_metric - second_operand

    assert isinstance(final_sub, CompositionalMetric)

    assert torch.allclose(expected_result, final_sub.compute())


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(3), torch.tensor(2.0)),
        (3, torch.tensor(2.0)),
        (3.0, torch.tensor(2.0)),
        (torch.tensor(3), torch.tensor(2.0)),
    ],
)
@pytest.mark.skipif(**_MARK_TORCH_LOWER_1_4)
def test_metrics_truediv(second_operand, expected_result):
    first_metric = DummyMetric(6)

    final_truediv = first_metric / second_operand

    assert isinstance(final_truediv, CompositionalMetric)

    assert torch.allclose(expected_result, final_truediv.compute())


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [(DummyMetric([1, 0, 3]), torch.tensor([-2, -2, 0])), (torch.tensor([1, 0, 3]), torch.tensor([-2, -2, 0]))],
)
def test_metrics_xor(second_operand, expected_result):
    first_metric = DummyMetric([-1, -2, 3])

    final_xor = first_metric ^ second_operand
    final_rxor = second_operand ^ first_metric

    assert isinstance(final_xor, CompositionalMetric)
    assert isinstance(final_rxor, CompositionalMetric)

    assert torch.allclose(expected_result, final_xor.compute())
    assert torch.allclose(expected_result, final_rxor.compute())


def test_metrics_abs():
    first_metric = DummyMetric(-1)

    final_abs = abs(first_metric)

    assert isinstance(final_abs, CompositionalMetric)

    assert torch.allclose(torch.tensor(1), final_abs.compute())


def test_metrics_invert():
    first_metric = DummyMetric(1)

    final_inverse = ~first_metric
    assert isinstance(final_inverse, CompositionalMetric)
    assert torch.allclose(torch.tensor(-2), final_inverse.compute())


def test_metrics_neg():
    first_metric = DummyMetric(1)

    final_neg = neg(first_metric)
    assert isinstance(final_neg, CompositionalMetric)
    assert torch.allclose(torch.tensor(-1), final_neg.compute())


def test_metrics_pos():
    first_metric = DummyMetric(-1)

    final_pos = pos(first_metric)
    assert isinstance(final_pos, CompositionalMetric)
    assert torch.allclose(torch.tensor(1), final_pos.compute())


def test_compositional_metrics_update():

    compos = DummyMetric(5) + DummyMetric(4)

    assert isinstance(compos, CompositionalMetric)
    compos.update()
    compos.update()
    compos.update()

    assert isinstance(compos.metric_a, DummyMetric)
    assert isinstance(compos.metric_b, DummyMetric)

    assert compos.metric_a._num_updates == 3
    assert compos.metric_b._num_updates == 3
