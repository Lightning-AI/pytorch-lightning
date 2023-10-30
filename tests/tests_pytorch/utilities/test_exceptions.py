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
from pathlib import Path

from lightning.pytorch.utilities.exceptions import _augment_message


def test_augment_message():
    # exception without args
    exception = Exception()
    _augment_message(exception, "", "new message")
    assert not exception.args

    # exception with one arg
    exception = Exception("Test message.")
    _augment_message(exception, "Test", "New Test message")
    assert exception.args == ("New Test message",)

    # pattern matching
    exception = Exception("Hello. Test message. Over!")
    _augment_message(exception, ".*Test.*Over.*", "New Test message")
    assert exception.args == ("New Test message",)

    # exception with multiple args
    exception = Exception("Message 1", "Message 2", "Message 3")
    _augment_message(exception, "Message 2", "New message 2")
    assert exception.args == ("Message 1", "New message 2", "Message 3")

    # exception with non-string args
    path = Path("foo.yaml")
    exception = FileNotFoundError("Message 1", path)
    _augment_message(exception, "Message 1", "New message 1")
    assert exception.args == ("New message 1", path)
