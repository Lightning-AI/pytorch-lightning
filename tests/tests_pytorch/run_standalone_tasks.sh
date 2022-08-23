#!/bin/bash
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
set -e
# THIS FILE ASSUMES IT IS RUN INSIDE THE tests/tests_pytorch DIRECTORY

if nvcc --version; then
    echo "Running profilers/test_profiler.py::test_pytorch_profiler_nested_emit_nvtx"
    nvprof --profile-from-start off -o trace_name.prof -- python -m coverage run --source pytorch_lightning --append -m pytest --no-header profilers/test_profiler.py::test_pytorch_profiler_nested_emit_nvtx
fi

# needs to run outside of `pytest`
echo "Running utilities/test_warnings.py"
python utilities/test_warnings.py

# test deadlock is properly handled with TorchElastic.
echo "Running plugins/environments/torch_elastic_deadlock.py"
LOGS=$(PL_RUN_STANDALONE_TESTS=1 PL_RECONCILE_PROCESS=1 python -m torch.distributed.run --nproc_per_node=2 --max_restarts 0 -m coverage run --source pytorch_lightning -a plugins/environments/torch_elastic_deadlock.py | grep "SUCCEEDED")
if [ -z "$LOGS" ]; then
    exit 1
fi

# test that a user can manually launch individual processes
echo "Running manual ddp launch test"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
args="fit --trainer.accelerator gpu --trainer.devices 2 --trainer.strategy ddp --trainer.max_epochs=1 --trainer.limit_train_batches=1 --trainer.limit_val_batches=1 --trainer.limit_test_batches=1"
MASTER_ADDR="localhost" MASTER_PORT=1234 LOCAL_RANK=1 python strategies/scripts/cli_script.py ${args} &
MASTER_ADDR="localhost" MASTER_PORT=1234 LOCAL_RANK=0 python strategies/scripts/cli_script.py ${args}

# test that ddp can launched as a module (-m option)
echo "Running ddp example as module"
python -m strategies.scripts.cli_script ${args}
