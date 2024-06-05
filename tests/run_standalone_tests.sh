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
source="${PL_STANDALONE_TESTS_SOURCE:-"lightning"}"

# this environment variable allows special tests to run
export PL_RUN_STANDALONE_TESTS=1
# python arguments
defaults=" -m coverage run --source ${source} --append -m pytest --no-header -v -s --timeout 120 "
echo "Using defaults: ${defaults}"

# get the testing location as the first argument
test_path=$1
printf "source path: $test_path\n"

# collect all tests with parametrization based filtering with PL_RUN_STANDALONE_TESTS
standalone_tests=$(python3 -m pytest $test_path -q --collect-only --pythonwarnings ignore)
printf "Collected tests: \n $standalone_tests\n"
# match only lines with tests
parametrizations=$(perl -nle 'print $& while m{\S+::test_\S+}g' <<< "$standalone_tests")
# convert the list to be array
parametrizations_arr=($parametrizations)
report=''

rm -f standalone_test_output.txt  # in case it exists, remove it
rm -f testnames.txt

function show_batched_output {
  if [ -f standalone_test_output.txt ]; then  # if exists
    cat standalone_test_output.txt
    # heuristic: stop if there's mentions of errors. this can prevent false negatives when only some of the ranks fail
    if perl -nle 'print if /error|(?<!(?-i)on_)exception|traceback|(?<!(?-i)x)failed/i' standalone_test_output.txt | grep -qv -f testnames.txt; then
      echo "Potential error! Stopping."
      rm standalone_test_output.txt
      exit 1
    fi
    rm standalone_test_output.txt
  fi
}
trap show_batched_output EXIT  # show the output on exit

# remove the "tests/tests_pytorch/" path suffixes
path_prefix=$(basename "$(dirname "$(pwd)")")/$(basename "$(pwd)")"/"  # https://stackoverflow.com/a/8223345

for i in "${!parametrizations_arr[@]}"; do
  parametrization=${parametrizations_arr[$i]//$path_prefix/}
  prefix="$((i+1))/${#parametrizations_arr[@]}"

  echo "$prefix: Running $parametrization"
  echo $parametrization | sed 's/\[[^][]*\]//g' >> testnames.txt

  # fix the port to avoid race condition when batched distributed tests select the port randomly
  export MASTER_PORT=$((29500 + $i % $test_batch_size))

  # execute the test in the background
  # redirect to a log file that buffers test output. since the tests will run in the background, we cannot let them
  # output to std{out,err} because the outputs would be garbled together
  python3 ${defaults} "$parametrization" &>> standalone_test_output.txt &
  # save the PID in an array
  pids[${i}]=$!
  # add row to the final report
  report+="Ran\t$parametrization\n"

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
