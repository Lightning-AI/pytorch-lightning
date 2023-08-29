from lightning_utilities.core.rank_zero import rank_zero_warn


class restricted_classmethod:
    """
    Custom `classmethod` that emits a warning when the classmethod is
    called on an instance and not the class type.
    """

    def __init__(self, method):
        self.method = method

    def __get__(self, instance, cls):
        if instance is not None:
            rank_zero_warn(
                f"The classmethod {cls.__name__}.{self.method.__name__} was called on an instance. Please "
                f"call this method on the class type ({cls.__name__}) and make sure the return value is used"
            )
        return lambda *args, **kwargs: self.method(cls, *args, **kwargs)
