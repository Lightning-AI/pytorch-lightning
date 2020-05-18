from pytorch_lightning.loggers import MLFlowLogger


def test_mlflow_logger_exists(tmpdir):
    """Verify that basic functionality of mlflow logger works."""
    logger = MLFlowLogger('test', save_dir=tmpdir)
    # Test already exists
    logger2 = MLFlowLogger('test', save_dir=tmpdir)
    assert logger.run_id != logger2.run_id
