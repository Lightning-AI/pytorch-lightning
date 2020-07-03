from enum import Enum, auto

from pytorch_lightning import Callback


class TrainerState(Enum):
    """ State which is set to the Trainer to indicate what is being executed. """
    INITIALIZE = auto()
    RUNNING = auto()
    FINISHED = auto()


class _TrainerStateSwitcher(Callback):
    """ Special callback used by the Trainer. This callback sets proper
        state to the trainer depending on what is being executed.
    """

    def on_init_start(self, trainer):
        trainer.state = TrainerState.INITIALIZE

    def on_init_end(self, trainer):
        trainer.state = TrainerState.INITIALIZE

    def setup(self, trainer, stage: str):
        trainer.state = TrainerState.RUNNING

    def teardown(self, trainer, stage: str):
        trainer.state = TrainerState.FINISHED

    def on_keyboard_interrupt(self, trainer, pl_module):
        trainer.state = TrainerState.FINISHED
