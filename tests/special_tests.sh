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

# this environment variable allows special tests to run
export PL_RUNNING_SPECIAL_TESTS=1
# python arguments
defaults='-m coverage run --source pytorch_lightning --append -m pytest --durations=0 --capture=no --disable-warnings'

# find tests marked as `@RunIf(special=True)`
grep_output=$(grep --recursive --line-number --word-regexp 'tests' 'benchmarks' --regexp 'special=True')
# file paths
files=$(echo "$grep_output" | cut -f1 -d:)
files_arr=($files)
# line numbers
linenos=$(echo "$grep_output" | cut -f2 -d:)
linenos_arr=($linenos)

# tests to skip - space separated
blocklist='test_pytorch_profiler_nested_emit_nvtx'
report=''

for i in "${!files_arr[@]}"; do
  file=${files_arr[$i]}
  lineno=${linenos_arr[$i]}

  # get code from `@RunIf(special=True)` line to EOF
  test_code=$(tail -n +"$lineno" "$file")

  # read line by line
  while read -r line; do
    # if it's a test
    if [[ $line == def\ test_* ]]; then
      # get the name
      test_name=$(echo $line | cut -c 5- | cut -f1 -d\()

      # check blocklist
      if echo $blocklist | grep --word-regexp "$test_name" > /dev/null; then
        report+="Skipped\t$file:$lineno::$test_name\n"
        break
      fi

      # SPECIAL_PATTERN allows filtering the tests to run when debugging.
      # use as `SPECIAL_PATTERN="foo_bar" ./special_tests.sh` to run only those
      # test with `foo_bar` in their name
      if [[ $line != *$SPECIAL_PATTERN* ]]; then
        report+="Skipped\t$file:$lineno::$test_name\n"
        break
      fi

      # run the test
      report+="Ran\t$file:$lineno::$test_name\n"
      python ${defaults} "${file}::${test_name}"
      break
    fi
  done < <(echo "$test_code")
done

if nvcc --version; then
    nvprof --profile-from-start off -o trace_name.prof -- python ${defaults} tests/profiler/test_profiler.py::test_pytorch_profiler_nested_emit_nvtx
fi

# needs to run outside of `pytest`
python tests/utilities/test_warnings.py
if [ $? -eq 0 ]; then
    report+="Ran\ttests/utilities/test_warnings.py\n"
fi

# TODO: enable when CI uses torch>=1.9
# test deadlock is properly handled with TorchElastic.
# LOGS=$(PL_RUNNING_SPECIAL_TESTS=1 python -m torch.distributed.run --nproc_per_node=2 --max_restarts 0 -m coverage run --source pytorch_lightning -a tests/plugins/environments/torch_elastic_deadlock.py | grep "SUCCEEDED")
# if  [ -z "$LOGS" ]; then
#    exit 1
# fi
# report+="Ran\ttests/plugins/environments/torch_elastic_deadlock.py\n"

# test that a user can manually launch individual processes
args="--trainer.gpus 2 --trainer.accelerator ddp --trainer.max_epochs=1 --trainer.limit_train_batches=1 --trainer.limit_val_batches=1 --trainer.limit_test_batches=1"
MASTER_ADDR="localhost" MASTER_PORT=1234 LOCAL_RANK=1 python pl_examples/basic_examples/simple_image_classifier.py ${args} &
MASTER_ADDR="localhost" MASTER_PORT=1234 LOCAL_RANK=0 python pl_examples/basic_examples/simple_image_classifier.py ${args}
report+="Ran\tmanual ddp launch test\n"

# echo test report
printf '=%.s' {1..80}
printf "\n$report"
printf '=%.s' {1..80}
printf '\n'
