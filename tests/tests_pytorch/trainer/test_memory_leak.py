import os
import psutil
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel


class CustomModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1000, 2)  # Changed to match LargeDataset dim=1000

    def forward(self, x):
        return self.layer(x)


class LargeDataset(Dataset):
    def __init__(self, size=1000, dim=1000):
        self.data = torch.randn(size, dim)
        self.targets = torch.randint(0, 10, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        # During prediction, return only the input tensor
        if hasattr(self, 'prediction_mode') and self.prediction_mode:
            return self.data[idx]
        return self.data[idx], self.targets[idx]

    def set_prediction_mode(self, mode=True):
        self.prediction_mode = mode


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


@pytest.mark.parametrize("return_predictions", [True, False])
def test_prediction_memory_leak(tmp_path, return_predictions):
    """Test that memory usage doesn't grow during prediction when return_predictions=False."""
    # Create a model and dataset
    model = CustomModel()
    dataset = LargeDataset()
    dataset.set_prediction_mode(True)  # Set prediction mode
    dataloader = DataLoader(dataset, batch_size=32)

    # Get initial memory usage
    initial_memory = get_memory_usage()

    # Run prediction
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_epochs=1,
    )
    
    predictions = trainer.predict(model, dataloaders=dataloader, return_predictions=return_predictions)
    
    # Get final memory usage
    final_memory = get_memory_usage()
    
    # Calculate memory growth
    memory_growth = final_memory - initial_memory
    
    # When return_predictions=False, memory growth should be minimal
    if not return_predictions:
        assert memory_growth < 100, f"Memory growth {memory_growth}MB is too high when return_predictions=False"
    else:
        # When return_predictions=True, we expect some memory growth due to storing predictions
        assert memory_growth > 0, "Expected memory growth when storing predictions" 