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
from lightning.pytorch.utilities.exceptions import _augment_message
from lightning.pytorch.utilities.imports import _PYTHON_GREATER_EQUAL_3_11_0


def test_augment_message():
    # exception without args
    exception = Exception()
    _augment_message(exception, "", "new message")
    assert not exception.args
    if _PYTHON_GREATER_EQUAL_3_11_0:
        assert not exception.__notes__

    # exception with one arg
    exception = Exception("Test message.")
    _augment_message(exception, "Test", "New Test message")
    if _PYTHON_GREATER_EQUAL_3_11_0:
        assert exception.__notes__ == ["New Test message"]
        assert exception.args == ("Test message.",)
    else:
        assert exception.args == ("New Test message",)

    # pattern matching
    exception = Exception("Hello. Test message. Over!")
    _augment_message(exception, ".*Test.*Over.*", "New Test message")
    if _PYTHON_GREATER_EQUAL_3_11_0:
        assert exception.__notes__ == ["New Test message"]
        assert exception.args == ("Hello. Test message. Over!",)
    else:
        assert exception.args == ("New Test message",)

    # exception with multiple args
    exception = Exception("Message 1", "Message 2", "Message 3")
    _augment_message(exception, "Message 2", "New message 2")
    if _PYTHON_GREATER_EQUAL_3_11_0:
        assert exception.__notes__ == ["New message 2"]
        assert exception.args == (
            "Message 1",
            "Message 2",
            "Message 3",
        )
    else:
        assert exception.args == ("Message 1", "New message 2", "Message 3")
