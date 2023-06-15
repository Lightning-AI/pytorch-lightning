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

# THIS FILE ASSUMES IT IS RUN INSIDE THE tests/parity_fabric DIRECTORY
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

MAX_RETRIES=3

retry_command() {
  local command="$@"
  local exit_code=1
  for ((i=1; i<=$MAX_RETRIES; i++))
  do
    echo "Run attempt: $i"
    eval $command
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
      echo "Successfully ran: $command"
      break
    fi
    echo "Attempt $i failed."
  done
  if [ $exit_code -ne 0 ]; then
    echo "Failed after $MAX_RETRIES attempts: $command"
  fi
  return $exit_code
}

retry_command "python -m test_parity_ddp --accelerator="cpu" --devices=2 --tolerance=0.02"
retry_command "python -m test_parity_ddp --accelerator="cuda" --devices=2 --tolerance=0.01"
