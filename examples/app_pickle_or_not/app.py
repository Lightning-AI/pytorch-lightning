import logging

import lightning as L

logger = logging.getLogger(__name__)


class PickleChecker(L.LightningWork):
    def run(self, pickle_image: bytes):
        parsed = self.parse_image(pickle_image)
        if parsed == b"it is a pickle":
            return True
        elif parsed == b"it is not a pickle":
            return False
        else:
            raise Exception("Couldn't parse the image")

    @staticmethod
    def parse_image(image_str: bytes):
        return image_str


class Slack(L.LightningFlow):
    def __init__(self):
        super().__init__()

    @staticmethod
    def send_message(message):
        logger.info(f"Sending message: {message}")

    def run(self):
        pass


class RootComponent(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.pickle_checker = PickleChecker()
        self.slack = Slack()
        self.counter = 3

    def run(self):
        if self.counter > 0:
            logger.info(f"Running the app {self.counter}")
            image_str = b"it is not a pickle"
            if self.pickle_checker.run(image_str):
                self.slack.send_message("It's a pickle!")
            else:
                self.slack.send_message("It's not a pickle!")
            self.counter -= 1
        else:
            self._exit("Pickle or Not End")


app = L.LightningApp(RootComponent())
