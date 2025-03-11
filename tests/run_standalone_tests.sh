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

# THIS FILE ASSUMES IT IS RUN INSIDE THE tests DIRECTORY.

# Batch size for testing: Determines how many standalone test invocations run in parallel
# It can be set through the env variable NUM_PARALLEL_TESTS and defaults to 5 if not set
test_batch_size="${NUM_PARALLEL_TESTS:-5}"

# Source directory for coverage runs can be set with CODECOV_SOURCE and defaults to lightning.
codecov_source="${COVERAGE_SOURCE:-"lightning"}"

# The test directory is passed as the first argument to the script
test_dir=$1 # parse the first argument

# There is also timeout for the tests.
# It can be set through the env variable TEST_TIMEOUT and defaults to 1200 seconds if not set 1200 seconds
test_timeout="${TEST_TIMEOUT:-1200}"

# Temporary file to store the collected tests
COLLECTED_TESTS_FILE="collected_tests.txt"

ls -lh .  # show the contents of the directory

# Python arguments for running the tests and coverage
defaults=" -m coverage run --source ${codecov_source} --append -m pytest --no-header -v -s --color=yes --timeout=${test_timeout} --durations=0 "
echo "Using defaults: ${defaults}"

# Get the list of parametrizations. we need to call them separately. the last two lines are removed.
# note: if there's a syntax error, this will fail with some garbled output
python -um pytest ${test_dir} -q --collect-only --pythonwarnings ignore 2>&1 > $COLLECTED_TESTS_FILE
# Early terminate if collection failed (e.g. syntax error)
if [[ $? != 0 ]]; then
  cat $COLLECTED_TESTS_FILE
  printf "ERROR: test collection failed!\n"
  exit 1
fi

# Initialize empty array
tests=()

# Read from file line by line
while IFS= read -r line; do
    # Only keep lines containing "test_"
    if [[ $line == *"test_"* ]]; then
        # Extract part after test_dir/
        pruned_line="${line#*${test_dir}/}"
        tests+=("${test_dir}/$pruned_line")
    fi
done < $COLLECTED_TESTS_FILE

# Count tests
test_count=${#tests[@]}

# Display results
printf "collected $test_count tests:\n-------------------\n"
printf "%s\n" "${tests[@]}"
printf "\n===================\n"

# if test count is one print warning
if [[ $test_count -eq 1 ]]; then
  printf "WARNING: only one test found!\n"
elif [ $test_count -eq 0 ]; then
  printf "ERROR: no tests found!\n"
  exit 1
fi

# clear all the collected reports
rm -f parallel_test_output-*.txt  # in case it exists, remove it

status=0 # aggregated script status
report="" # final report
pids=() # array of PID for running tests
test_ids=() # array of indexes of running tests
failed_tests=() # array of failed tests
printf "Running $test_count tests in batches of $test_batch_size:\n"
for i in "${!tests[@]}"; do
  test=${tests[$i]}
  printf "* Running test $((i+1))/$test_count: $test\n"

  # execute the test in the background
  # redirect to a log file that buffers test output. since the tests will run in the background,
  # we cannot let them output to std{out,err} because the outputs would be garbled together
  python ${defaults} "$test" &> "parallel_test_output-$i.txt" &
  test_ids+=($i) # save the test's id in an array with running tests
  pids+=($!) # save the PID in an array with running tests

  # if we reached the batch size, wait for all tests to finish
  if (( (($i + 1) % $test_batch_size == 0) || $i == $test_count-1 )); then
    printf "-> Waiting for batch to finish: $(IFS=' '; echo "${pids[@]}")\n"
    # wait for running tests
    for j in "${!test_ids[@]}"; do
      i=${test_ids[$j]} # restore the global test's id
      pid=${pids[$j]} # restore the particular PID
      test=${tests[$i]} # restore the test name
      printf "? Waiting for $tests >> parallel_test_output-$i.txt (PID: $pid)\n"
      wait -n $pid
      # get the exit status of the test
      test_status=$?
      # add row to the final report
      report+="Ran\t$test\t>> exit:$test_status\n"
      if [[ $test_status != 0 ]]; then
        # add the test to the failed tests array
        failed_tests+=($i)
        # Process exited with a non-zero exit status
        status=$test_status
      fi
    done
    printf "Starting over with a new batch...\n"
    test_ids=()  # reset the test's id array
    pids=()  # reset the PID array
  fi
done

# print test report with exit code for each test
printf '=%.s' {1..80}
printf "\n$report"
printf '=%.s' {1..80}
printf '\n'

# print failed tests from duped logs
if [[ ${#failed_tests[@]} -gt 0 ]]; then
  printf "Failed tests:\n"
  for i in "${failed_tests[@]}"; do
    printf '\n%.s' {1..5}
    printf '=%.s' {1..80}
    printf "\n${tests[$i]}\n"
    printf '-%.s' {1..80}
    printf "\n"
    # show the output of the failed test
    cat "parallel_test_output-$i.txt"
    printf "\n"
    printf '=%.s' {1..80}
  done
else
  printf "All tests passed!\n"
fi

# exit with the worse test result
exit $status
