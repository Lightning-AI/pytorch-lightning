LOGS=$(PL_RUNNING_SPECIAL_TESTS=1 python -m torch.distributed.launch --nproc_per_node=2  tests/plugins/environments/torch_elastic_deadlock.py | grep "SUCCEEDED")
if  [ -z "$LOGS" ]; then
    exit 1
fi