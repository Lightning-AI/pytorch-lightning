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

# Source directory for coverage runs can be set with CODECOV_SOURCE.
codecov_source="${COVERAGE_SOURCE}"

# The test directory is passed as the first argument to the script
test_dir=$1 # parse the first argument

# There is also timeout for the tests.
# It can be set through the env variable TEST_TIMEOUT and defaults to 1200 seconds.
test_timeout="${TEST_TIMEOUT:-1200}"

# Temporary file to store the collected tests
COLLECTED_TESTS_FILE="collected_tests.txt"

ls -lh .  # show the contents of the directory

# If codecov_source is set, prepend the coverage command
if [ -n "$codecov_source" ]; then
  cli_coverage="-m coverage run --source ${codecov_source} --append"
else # If not, just keep it empty
  cli_coverage=""
fi
# Append the common pytest arguments
cli_pytest="-m pytest --no-header -v -s --color=yes --timeout=${test_timeout}"

# Python arguments for running the tests and optional coverage
printf "\e[35mUsing defaults: ${cli_coverage} ${cli_pytest}\e[0m\n"

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
printf "\e[34m================================================================================\e[0m\n"
printf "\e[34mCOLLECTED $test_count TESTS:\e[0m\n"
printf "\e[34m~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\e[0m\n"
printf "\e[34m%s\e[0m\n" "${tests[@]}"
printf "\e[34m================================================================================\e[0m\n"

# if test count is one print warning
if [[ $test_count -eq 1 ]]; then
  printf "\e[33mWARNING: only one test found!\e[0m\n"
elif [ $test_count -eq 0 ]; then
  printf "\e[31mERROR: no tests found!\e[0m\n"
  exit 1
fi

if [ -n "$codecov_source" ]; then
  coverage combine
fi

status=0 # aggregated script status
report=() # final report
pids=() # array of PID for running tests
test_ids=() # array of indexes of running tests
failed_tests=() # array of failed tests
printf "Running $test_count tests in batches of $test_batch_size:\n"
for i in "${!tests[@]}"; do
  test=${tests[$i]}

  cli_test="python "
  if [ -n "$codecov_source" ]; then
    # append cli_coverage to the test command
    cli_test="${cli_test} ${cli_coverage} --data-file=run-${i}.coverage"
  fi
  # add the pytest cli to the test command
  cli_test="${cli_test} ${cli_pytest}"

  printf "\e[95m* Running test $((i+1))/$test_count: $cli_test $test\e[0m\n"

  # execute the test in the background
  # redirect to a log file that buffers test output. since the tests will run in the background,
  # we cannot let them output to std{out,err} because the outputs would be garbled together
  ${cli_test} "$test" &> "parallel_test_output-$i.txt" &
  test_ids+=($i) # save the test's id in an array with running tests
  pids+=($!) # save the PID in an array with running tests

  # if we reached the batch size, wait for all tests to finish
  if (( (($i + 1) % $test_batch_size == 0) || $i == $test_count-1 )); then
    printf "Waiting for batch to finish: $(IFS=' '; echo "${pids[@]}")\n"
    # wait for running tests
    for j in "${!test_ids[@]}"; do
      i=${test_ids[$j]} # restore the global test's id
      pid=${pids[$j]} # restore the particular PID
      test=${tests[$i]} # restore the test name
      printf "\e[33m? Waiting for $test @ parallel_test_output-$i.txt (PID: $pid)\e[0m\n"
      wait -n $pid
      # get the exit status of the test
      test_status=$?
      # add row to the final report
      report+=("Ran $test >> exit:$test_status")
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
printf "\e[35m================================================================================\e[0m\n"
for line in "${report[@]}"; do
    if [[ "$line" == *"exit:0"* ]]; then
        printf "\e[32m%s\e[0m\n" "$line"  # Green for lines containing exit:0
    else
        printf "\e[31m%s\e[0m\n" "$line" # Red for all other lines
    fi
done
printf "\e[35m================================================================================\e[0m\n"

# print failed tests from duped logs
if [[ ${#failed_tests[@]} -gt 0 ]]; then
  printf "\e[34mFAILED TESTS:\e[0m\n"
  for i in "${failed_tests[@]}"; do
    printf "\e[34m================================================================================\e[0m\n"
    printf "\e[34m=== ${tests[$i]} ===\e[0m\n"
    printf "\e[34m~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\e[0m\n\n"
    # show the output of the failed test
    cat "parallel_test_output-$i.txt"
    printf "\e[34m~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\e[0m\n"
    printf "\e[34m================================================================================\e[0m\n"
    printf '\n\n\n'
  done
else
  printf "\e[32mAll tests passed!\e[0m\n"
fi

# exit with the worse test result
exit $status
