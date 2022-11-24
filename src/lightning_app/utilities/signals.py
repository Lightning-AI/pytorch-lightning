class SignalHandlerCompose:

    """This class enables to compose dynamically signal handlers.

    Example:
        import signal

        signal.signal = SignalHandlerCompose(signal.signal)
    """

    def __init__(self, original_signal):
        self.original_signal = original_signal
        self.signums = {}

    def run(self, signum, frame):
        for callable in self.signums[signum]:
            callable(signum, frame)

    def __call__(self, signum, callable):
        if signum not in self.signums:
            self.signums[signum] = []

        self.signums[signum].append(callable)

        self.original_signal(signum, self.run)
