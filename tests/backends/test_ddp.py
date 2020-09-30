import pytest
import torch
import os
from tests.backends import ddp_model
from tests.utilities.dist import call_training_script


@pytest.mark.parametrize('cli_args', [
    pytest.param('--max_epochs 1 --gpus 2 --distributed_backend ddp'),
])
# @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_multi_gpu_model_ddp_fit_only(tmpdir, cli_args):
    # call the script
    std, err = call_training_script(ddp_model, cli_args, 'fit', tmpdir, timeout=120)
    print(std)
    print(err)

    # load the results of the script
    result_path = os.path.join(tmpdir, 'ddp.result')
    result = torch.load(result_path)

    # verify the file wrote the expected outputs
    assert result['status'] == 'complete'


# @pytest.mark.parametrize('cli_args', [
#     pytest.param('--max_epochs 1 --gpus 2 --distributed_backend ddp'),
# ])
# @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
# def test_multi_gpu_model_ddp_test_only(tmpdir, cli_args):
#     # call the script
#     call_training_script(ddp_model, cli_args, 'test', tmpdir)
#
#     # load the results of the script
#     result_path = os.path.join(tmpdir, 'ddp.result')
#     result = torch.load(result_path)
#
#     # verify the file wrote the expected outputs
#     assert result['status'] == 'complete'


# @pytest.mark.parametrize('cli_args', [
#     pytest.param('--max_epochs 1 --gpus 2 --distributed_backend ddp'),
# ])
# @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
# def test_multi_gpu_model_ddp_fit_test_only(tmpdir, cli_args):
#     # call the script
#     call_training_script(ddp_model, cli_args, 'fit_test', tmpdir, timeout=20)
#
#     # load the results of the script
#     result_path = os.path.join(tmpdir, 'ddp.result')
#     result = torch.load(result_path)
#
#     # verify the file wrote the expected outputs
#     assert result['status'] == 'complete'
#
#     model_outs = result['result']
#     for out in model_outs:
#         assert out['test_acc'] > 0.90

#
#
# @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
# def test_multi_gpu_early_stop_ddp_spawn(tmpdir):
#     """Make sure DDP works. with early stopping"""
#     tutils.set_random_master_port()
#
#     trainer_options = dict(
#         default_root_dir=tmpdir,
#         early_stop_callback=True,
#         max_epochs=50,
#         limit_train_batches=10,
#         limit_val_batches=10,
#         gpus=[0, 1],
#         distributed_backend='ddp_spawn',
#     )
#
#     model = EvalModelTemplate()
#     tpipes.run_model_test(trainer_options, model)
