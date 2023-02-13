#!/bin/bash
# Copyright The Lightning AI team.
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
# THIS FILE ASSUMES IT IS RUN INSIDE THE tests/tests_<package> DIRECTORY

# Batch size for testing: Determines how many standalone test invocations run in parallel
# It can be set through the env variable PL_STANDALONE_TESTS_BATCH_SIZE and defaults to 6 if not set
test_batch_size="${PL_STANDALONE_TESTS_BATCH_SIZE:-6}"
source="${PL_STANDALONE_TESTS_SOURCE:-"lightning.pytorch"}"

# this environment variable allows special tests to run
export PL_RUN_STANDALONE_TESTS=1
# python arguments
defaults="-m coverage run --source $source --append -m pytest --no-header -v -s"

# find tests marked as `@RunIf(standalone=True)`. done manually instead of with pytest because it is faster
grep_output=$(grep --recursive --word-regexp . --regexp 'standalone=True' --include '*.py')

# file paths, remove duplicates
files=$(echo "$grep_output" | cut -f1 -d: | sort | uniq)

# get the list of parametrizations. we need to call them separately. the last two lines are removed.
# note: if there's a syntax error, this will fail with some garbled output
if [[ "$OSTYPE" == "darwin"* ]]; then
  parametrizations=$(python -m pytest $files --collect-only --quiet "$@" | tail -r | sed -e '1,3d' | tail -r)
else
  parametrizations=$(python -m pytest $files --collect-only --quiet "$@" | head -n -2)
fi
# remove the "tests/tests_pytorch/" path suffixes
path_suffix=$(basename "$(dirname "$(pwd)")")/$(basename "$(pwd)")"/"  # https://stackoverflow.com/a/8223345
parametrizations=${parametrizations//$path_suffix/}
parametrizations_arr=($parametrizations)

report=''

rm -f standalone_test_output.txt  # in case it exists, remove it
function show_batched_output {
  if [ -f standalone_test_output.txt ]; then  # if exists
    cat standalone_test_output.txt
    # heuristic: stop if there's mentions of errors. this can prevent false negatives when only some of the ranks fail
    if grep -iE 'error|exception|traceback|failed' standalone_test_output.txt | grep -qv 'on_exception'; then
      echo "Potential error! Stopping."
      rm standalone_test_output.txt
      exit 1
    fi
    rm standalone_test_output.txt
  fi
}
trap show_batched_output EXIT  # show the output on exit

for i in "${!parametrizations_arr[@]}"; do
  parametrization=${parametrizations_arr[$i]}

  # check blocklist
  if [[ "${parametrization}" == *"test_pytorch_profiler_nested_emit_nvtx"* ]]; then
    echo "Skipping $parametrization"
    report+="Skipped\t$parametrization\n"
    # do not continue the loop because we might need to wait for batched jobs
  else
    echo "Running $parametrization"
    # execute the test in the background
    # redirect to a log file that buffers test output. since the tests will run in the background, we cannot let them
    # output to std{out,err} because the outputs would be garbled together
    python ${defaults} "$parametrization" &>> standalone_test_output.txt &
    # save the PID in an array
    pids[${i}]=$!
    # add row to the final report
    report+="Ran\t$parametrization\n"
  fi

  if ((($i + 1) % $test_batch_size == 0)); then
    # wait for running tests
    for pid in ${pids[*]}; do wait $pid; done
    unset pids  # empty the array
    show_batched_output
  fi
done
# wait for leftover tests
for pid in ${pids[*]}; do wait $pid; done
show_batched_output

# echo test report
printf '=%.s' {1..80}
printf "\n$report"
printf '=%.s' {1..80}
printf '\n'
