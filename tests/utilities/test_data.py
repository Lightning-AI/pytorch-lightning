import torch

from pytorch_lightning.utilities.data import extract_batch_size


def test_extract_batch_size():
    """Tests the behavior of extracting the batch size."""
    batch = "test string"
    assert extract_batch_size(batch) == 11

    batch = torch.zeros(11, 10, 9, 8)
    assert extract_batch_size(batch) == 11

    batch = {"test": torch.zeros(11, 10)}
    assert extract_batch_size(batch) == 11

    batch = [torch.zeros(11, 10)]
    assert extract_batch_size(batch) == 11

    batch = {"test": [{"test": [torch.zeros(11, 10)]}]}
    assert extract_batch_size(batch) == 11
