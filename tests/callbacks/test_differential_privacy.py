import math
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import DifferentialPrivacy
from tests.base import EvalModelTemplate


def test_differential_privacy_accuracy(tmpdir):
    """Test to check the accuracy with DP is similar to without DP"""
    os.environ["PL_DEV_DEBUG"] = "1"
    seed_everything(42)

    # Train with Differential Privacy
    model = EvalModelTemplate(hidden_dim=1024)

    expected_count = 4
    trainer_with_dp = Trainer(
        default_root_dir=tmpdir,
        differential_privacy_callback=DifferentialPrivacy(noise_multiplier=0.3, max_grad_norm=0.1),
        max_epochs=expected_count,
        deterministic=True,
    )
    trainer_with_dp.fit(model)
    result_with_dp = trainer_with_dp.test()

    # Train without Differential Privacy
    model = EvalModelTemplate()
    expected_count = 4
    trainer_no_dp = Trainer(default_root_dir=tmpdir, max_epochs=expected_count, deterministic=True)
    trainer_no_dp.fit(model)
    result_no_dp = trainer_no_dp.test()

    # check is results are close
    dp_test_acc_ = result_with_dp[0]["test_acc"]
    no_dp_test_acc_ = result_no_dp[0]["test_acc"]
    assert math.isclose(dp_test_acc_, no_dp_test_acc_, rel_tol=1e-1)


# TODO: Validation test
# TODO: works with tpu
# TODO: check that false and true also works
