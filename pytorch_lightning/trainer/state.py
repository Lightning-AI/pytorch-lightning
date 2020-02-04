import enum


class TrainerMode(enum.Enum):
    TRAINING = enum.auto()
    VALIDATING = enum.auto()
    TESTING = enum.auto()
