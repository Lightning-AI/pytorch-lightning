from typing import Dict, Optional, Union

from pydantic import BaseModel, Field


class FitProgress(BaseModel):
    current_epoch: int = 0
    train_batch_idx: int = 0
    total_train_batches: int = 0
    val_dataloader_idx: int = 0
    val_batch_idx: int = 0
    total_val_batches: int = 0
    global_step: int = 0
    estimated_stepping_batches: int = 0


class ValidateProgress(BaseModel):
    dataloader_idx: int = 0
    val_batch_idx: int = 0
    total_val_batches: int = 0


class TestProgress(BaseModel):
    dataloader_idx: int = 0
    test_batch_idx: int = 0
    total_test_batches: int = 0


class PredictProgress(BaseModel):
    dataloader_idx: int = 0
    predict_batch_idx: int = 0
    total_predict_batches: int = 0


class ProgressBarState(BaseModel):
    fit: FitProgress = Field(default_factory=FitProgress)
    val: ValidateProgress = Field(alias="validate", default_factory=ValidateProgress)
    test: TestProgress = Field(default_factory=TestProgress)
    predict: PredictProgress = Field(default_factory=PredictProgress)
    metrics: Dict[str, Union[float, str]] = {}


class TrainerState(BaseModel):
    fn: Optional[str] = None
    stage: Optional[str] = None
