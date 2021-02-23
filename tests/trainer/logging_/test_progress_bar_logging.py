import pytest

from pytorch_lightning import Trainer
from tests.helpers import BoringModel


def test_logging_to_progress_bar_with_reserved_key(tmpdir):
    """ Test that logging a metric with a reserved name to the progress bar raises a warning. """

    class TestModel(BoringModel):

        def training_step(self, *args, **kwargs):
            output = super().training_step(*args, **kwargs)
            self.log("loss", output["loss"], prog_bar=True)
            return output

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=2,
    )
    with pytest.warns(UserWarning, match="The progress bar already tracks a metric with the .* 'loss'"):
        trainer.fit(model)
