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
from threading import Thread

class ThreadExceptionHandler(Thread):
    """
    Adopted from https://stackoverflow.com/a/67022927
    """
    def __init__(self, target, args=(), kwargs={}):
        Thread.__init__(self, target=target, args=args, kwargs=kwargs)
        self.exception = None
    
    def run(self):
        try:
            self._target(*self._args, **self._kwargs)
        except Exception as e:
            self.exception = e

    def join(self):
        super().join()
        if self.exception:
            raise self.exception