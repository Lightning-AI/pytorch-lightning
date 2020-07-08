from enum import Enum


class TrainerState(Enum):
    """ State which is set in the Trainer to indicate what is currently or was executed. """
    INITIALIZE = 'INITIALIZE'
    RUNNING = 'RUNNING'
    FINISHED = 'FINISHED'
    INTERRUPTED = 'INTERRUPTED'
