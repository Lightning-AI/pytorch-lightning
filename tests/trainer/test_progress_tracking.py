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
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.progress import FitLoopProgress, TrainingLoopProgress
from tests.helpers import BoringModel


def test_progress_tracking_with_1_optimizer(tmpdir):

    class TestModel(BoringModel):
        pass

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=3,
        limit_val_batches=0,
        limit_test_batches=0,
    )

    trainer.fit(model)

    assert isinstance(trainer.fit_loop.progress_tracker, FitLoopProgress)
    assert isinstance(trainer.fit_loop.training_loop.progress_tracker, TrainingLoopProgress)
    assert trainer.fit_loop.training_loop.progress_tracker == trainer.fit_loop.progress_tracker.train

    progress_tracker = trainer.fit_loop.training_loop.progress_tracker
    assert progress_tracker.epoch.total.ready == 2
    assert progress_tracker.epoch.total.started == 2
    assert progress_tracker.epoch.total.processed == 2
    assert progress_tracker.epoch.total.completed == 2

    assert progress_tracker.epoch.current.ready == 1
    assert progress_tracker.epoch.current.started == 1
    assert progress_tracker.epoch.current.processed == 1
    assert progress_tracker.epoch.current.completed == 1

    assert progress_tracker.batch.total.ready == 6
    assert progress_tracker.batch.total.started == 6
    assert progress_tracker.batch.total.processed == 6
    assert progress_tracker.batch.total.completed == 6

    assert progress_tracker.batch.current.ready == 3
    assert progress_tracker.batch.current.started == 3
    assert progress_tracker.batch.current.processed == 3
    assert progress_tracker.batch.current.completed == 3

    assert progress_tracker.optimizations[0].optimizer.total.ready == 6
    assert progress_tracker.optimizations[0].optimizer.total.started == 6
    assert progress_tracker.optimizations[0].optimizer.total.processed is None
    assert progress_tracker.optimizations[0].optimizer.total.completed == 6

    assert progress_tracker.optimizations[0].optimizer.current.ready == 1
    assert progress_tracker.optimizations[0].optimizer.current.started == 1
    assert progress_tracker.optimizations[0].optimizer.current.processed is None
    assert progress_tracker.optimizations[0].optimizer.current.completed == 1

    assert progress_tracker.optimizer_idx == 0


def test_progress_tracking_with_3_optimizers(tmpdir):

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx, optimizer_idx):
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            optimizer, lr_scheduler = super().configure_optimizers()
            optimizer, lr_scheduler = optimizer[0], lr_scheduler[0]
            return [optimizer, optimizer, optimizer], [lr_scheduler, lr_scheduler, lr_scheduler]

    model = TestModel()
    model.training_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=3,
        limit_val_batches=0,
        limit_test_batches=0,
    )

    trainer.fit(model)

    assert isinstance(trainer.fit_loop.progress_tracker, FitLoopProgress)
    assert isinstance(trainer.fit_loop.training_loop.progress_tracker, TrainingLoopProgress)
    assert trainer.fit_loop.training_loop.progress_tracker == trainer.fit_loop.progress_tracker.train

    progress_tracker = trainer.fit_loop.training_loop.progress_tracker
    assert progress_tracker.epoch.total.ready == 2
    assert progress_tracker.epoch.total.started == 2
    assert progress_tracker.epoch.total.processed == 2
    assert progress_tracker.epoch.total.completed == 2

    assert progress_tracker.epoch.current.ready == 1
    assert progress_tracker.epoch.current.started == 1
    assert progress_tracker.epoch.current.processed == 1
    assert progress_tracker.epoch.current.completed == 1

    assert progress_tracker.batch.total.ready == 6
    assert progress_tracker.batch.total.started == 6
    assert progress_tracker.batch.total.processed == 6
    assert progress_tracker.batch.total.completed == 6

    assert progress_tracker.batch.current.ready == 3
    assert progress_tracker.batch.current.started == 3
    assert progress_tracker.batch.current.processed == 3
    assert progress_tracker.batch.current.completed == 3

    assert progress_tracker.optimizations[0].optimizer.total.ready == 18
    assert progress_tracker.optimizations[0].optimizer.total.started == 18
    assert progress_tracker.optimizations[0].optimizer.total.processed is None
    assert progress_tracker.optimizations[0].optimizer.total.completed == 18

    assert progress_tracker.optimizations[0].optimizer.current.ready == 3
    assert progress_tracker.optimizations[0].optimizer.current.started == 3
    assert progress_tracker.optimizations[0].optimizer.current.processed is None
    assert progress_tracker.optimizations[0].optimizer.current.completed == 3

    assert progress_tracker.optimizer_idx == 2
