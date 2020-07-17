from enum import Enum
from functools import wraps
from typing import Callable, Optional

import pytorch_lightning


class TrainerState(Enum):
    """ State which is set in the Trainer to indicate what is currently or was executed. """
    INITIALIZE = 'INITIALIZE'
    RUNNING = 'RUNNING'
    FINISHED = 'FINISHED'
    INTERRUPTED = 'INTERRUPTED'


def trainer_state(*, entering: Optional[TrainerState] = None, exiting: Optional[TrainerState] = None) -> Callable:
    """ Decorator for :class:`~pytorch_lightning.Trainer` methods which changes
    state to `entering` before the function execution and `exiting` after
    the function is executed. If None is passed the state is not changed.
    """

    def wrapper(fn) -> Callable:
        @wraps(fn)
        def wrapped_fn(self, *args, **kwargs):
            if not isinstance(self, pytorch_lightning.Trainer):
                return fn(self, *args, **kwargs)

            if entering is not None:
                self.state = entering
            result = fn(self, *args, **kwargs)

            # The INTERRUPTED state can be set inside the run function. To indicate that run was interrupted
            # we retain INTERRUPTED state
            if exiting is not None and self.state != TrainerState.INTERRUPTED:
                self.state = exiting
            return result

        return wrapped_fn

    return wrapper
