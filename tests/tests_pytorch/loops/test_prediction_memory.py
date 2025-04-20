import gc
import os
import psutil
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel


class LargeMemoryDataset(Dataset):
    def __init__(self, size=100, data_size=100000):
        self.size = size
        self.data_size = data_size
        self.data = [torch.randn(data_size) for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class MemoryTestModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.predictions = []

    def predict_step(self, batch, batch_idx):
        # Simulate large memory usage
        result = batch * 2
        if not self.trainer.predict_loop.return_predictions:
            # Clear memory if not returning predictions
            gc.collect()
        return result

    def predict_dataloader(self):
        return DataLoader(LargeMemoryDataset(), batch_size=16)


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment variables after each test."""
    env_backup = os.environ.copy()
    yield
    # Clean up environment variables
    os.environ.clear()
    os.environ.update(env_backup)


@pytest.mark.parametrize("return_predictions", [True, False])
def test_prediction_memory_usage(return_predictions):
    """Test that memory usage doesn't grow unbounded during prediction."""
    # Skip if running on TPU
    if os.environ.get("TPU_ML_PLATFORM"):
        pytest.skip("Test not supported on TPU platform")
        
    model = MemoryTestModel()
    trainer = Trainer(accelerator="cpu", devices=1, max_epochs=1)
    
    # Get initial memory usage
    initial_memory = get_memory_usage()
    
    # Run prediction
    predictions = trainer.predict(model, return_predictions=return_predictions)
    
    # Get final memory usage
    final_memory = get_memory_usage()
    
    # Calculate memory growth
    memory_growth = final_memory - initial_memory
    
    # If return_predictions is False, memory growth should be minimal
    if not return_predictions:
        assert memory_growth < 500, f"Memory growth {memory_growth}MB exceeds threshold"
    else:
        # With return_predictions=True, some memory growth is expected
        assert memory_growth > 0, "Expected some memory growth with return_predictions=True"


def test_prediction_memory_with_gc():
    """Test that memory usage stays constant when using gc.collect()."""
    # Skip if running on TPU
    if os.environ.get("TPU_ML_PLATFORM"):
        pytest.skip("Test not supported on TPU platform")
        
    model = MemoryTestModel()
    trainer = Trainer(accelerator="cpu", devices=1, max_epochs=1)
    
    # Get initial memory usage
    initial_memory = get_memory_usage()
    
    # Run prediction with gc.collect()
    trainer.predict(model, return_predictions=False)
    
    # Get final memory usage
    final_memory = get_memory_usage()
    
    # Memory growth should be minimal
    memory_growth = final_memory - initial_memory
    assert memory_growth < 500, f"Memory growth {memory_growth}MB exceeds threshold" 