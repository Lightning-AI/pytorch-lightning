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

import abc


class BaseType(abc.ABCMeta):
    """Base class for Types."""

    @abc.abstractmethod
    def serialize(self, data):  # pragma: no cover
        """Serialize the incoming data to send it through the network."""

    @abc.abstractmethod
    def deserialize(self, *args, **kwargs):  # pragma: no cover
        """Take the inputs from the network and deserilize/convert them them.

        Output from this method will go to the exposed method as arguments.
        """
