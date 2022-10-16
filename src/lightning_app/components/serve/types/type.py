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
