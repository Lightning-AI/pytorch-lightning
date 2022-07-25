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

report=''

if nvcc --version; then
    nvprof --profile-from-start off -o trace_name.prof -- python ${defaults} profilers/test_profiler.py::test_pytorch_profiler_nested_emit_nvtx
fi

# needs to run outside of `pytest`
python utilities/test_warnings.py
if [ $? -eq 0 ]; then
    report+="Ran\tutilities/test_warnings.py\n"
fi

# test deadlock is properly handled with TorchElastic.
LOGS=$(PL_RUN_STANDALONE_TESTS=1 PL_RECONCILE_PROCESS=1 python -m torch.distributed.run --nproc_per_node=2 --max_restarts 0 -m coverage run --source pytorch_lightning -a plugins/environments/torch_elastic_deadlock.py | grep "SUCCEEDED")
if [ -z "$LOGS" ]; then
    exit 1
fi
report+="Ran\tplugins/environments/torch_elastic_deadlock.py\n"

# test that a user can manually launch individual processes
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
args="--trainer.accelerator gpu --trainer.devices 2 --trainer.strategy ddp --trainer.max_epochs=1 --trainer.limit_train_batches=1 --trainer.limit_val_batches=1 --trainer.limit_test_batches=1"
MASTER_ADDR="localhost" MASTER_PORT=1234 LOCAL_RANK=1 python ../../examples/convert_from_pt_to_pl/image_classifier_5_lightning_datamodule.py ${args} &
MASTER_ADDR="localhost" MASTER_PORT=1234 LOCAL_RANK=0 python ../../examples/convert_from_pt_to_pl/image_classifier_5_lightning_datamodule.py ${args}
report+="Ran\tmanual ddp launch test\n"

# echo test report
printf '=%.s' {1..80}
printf "\n$report"
printf '=%.s' {1..80}
printf '\n'