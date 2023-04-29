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

# THIS FILE ASSUMES IT IS RUN INSIDE THE tests/tests_fabric DIRECTORY
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

echo "Run parity tests manually"

for i in {1..3}
do
  echo "Run attempt: $i"
  python -m parity.test_parity_ddp --accelerator="cpu" --devices=2 --tolerance=0.02
  if [ $? -eq 0 ]; then break; fi
done


for i in {1..3}
do
  echo "Run attempt: $i"
  python -m parity.test_parity_ddp --accelerator="cuda" --devices=2 --tolerance=0.01
  if [ $? -eq 0 ]; then break; fi
done
