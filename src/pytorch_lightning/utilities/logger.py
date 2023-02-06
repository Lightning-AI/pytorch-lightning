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
"""Utilities for loggers."""

from pathlib import Path
from typing import Any, List, Tuple, Union

from pytorch_lightning.callbacks import Checkpoint


def _version(loggers: List[Any], separator: str = "_") -> Union[int, str]:
    if len(loggers) == 1:
        return loggers[0].version
    else:
        # Concatenate versions together, removing duplicates and preserving order
        return separator.join(dict.fromkeys(str(logger.version) for logger in loggers))


def _scan_checkpoints(checkpoint_callback: Checkpoint, logged_model_time: dict) -> List[Tuple[float, str, float, str]]:
    """Return the checkpoints to be logged.

    Args:
        checkpoint_callback: Checkpoint callback reference.
        logged_model_time: dictionary containing the logged model times.
    """

    # get checkpoints to be saved with associated score
    checkpoints = dict()
    if hasattr(checkpoint_callback, "last_model_path") and hasattr(checkpoint_callback, "current_score"):
        checkpoints[checkpoint_callback.last_model_path] = (checkpoint_callback.current_score, "latest")

    if hasattr(checkpoint_callback, "best_model_path") and hasattr(checkpoint_callback, "best_model_score"):
        checkpoints[checkpoint_callback.best_model_path] = (checkpoint_callback.best_model_score, "best")

    if hasattr(checkpoint_callback, "best_k_models"):
        for key, value in checkpoint_callback.best_k_models.items():
            checkpoints[key] = (value, "best_k")

    checkpoints = sorted(
        (Path(p).stat().st_mtime, p, s, tag) for p, (s, tag) in checkpoints.items() if Path(p).is_file()
    )
    checkpoints = [c for c in checkpoints if c[1] not in logged_model_time.keys() or logged_model_time[c[1]] < c[0]]
    return checkpoints
