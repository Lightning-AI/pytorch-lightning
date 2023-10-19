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

from typing import Any, List, TypeVar

T = TypeVar("T")


class DatasetOptimizationMixin:
    @staticmethod
    def prepare_dataset_structure(root: str, filepaths: List[str]) -> List[T]:
        """This function is meant to return a list of item metadata. Each item metadata should be enough to prepare a
        single item when called with the prepare_item.

        Example::

            # For a classification use case

            def prepare_dataset_structure(self, src_dir, filepaths)
                import numpy as np

                filepaths = ['class_a/file_1.ext', ..., 'class_b/file_1.ext', ...]
                classes = np.unique([filepath.split("/")[0] for filepath in filepaths])
                classes_to_idx_map = {c: idx for idx, c in enumerate(classes)}

                # Return pair with the filepath to the obj and its class
                # [('class_a/file_1.ext', 0), ... ('class_b/file_1.ext', 1)]
                return [(filepath, classes_to_idx_map[filepath.split("/")[0]]) for filepath in filepaths]

        Example::

            # For a image segmentation use case

            def prepare_dataset_structure(self, src_dir, filepaths)
                import numpy as np

                filepaths = ['file_1.JPEG', 'file_1.mask', .... 'file_N.JPEG', 'file_N.mask', ...]

                # [('file_1.JPEG', 'file_1.mask'), ... ('file_N.JPEG', 'file_N.mask')]
                return [(x[i], x[i+1]) for i in range(len(filepaths) -1)]

            def prepare_item(self, obj):
                image_filepath, mask_filepath = obj

                image = load_and_resize(image_filepath)
                mask = load_and_resize(mask_filepath)
                return (image, mask)

        """

    @staticmethod
    def prepare_item(item_metadata: T) -> Any:
        """Using some metadata, prepare the associated item.

        The output of this function will be binarised

        """
        return item_metadata
