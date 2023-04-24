import logging
import sys
import time

from lightning.app import LightningApp, LightningFlow, LightningWork

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class SimpleWork(LightningWork):
    def __init__(self):
        super().__init__(cache_calls=False, parallel=True, raise_exception=False)
        self.is_running_now = False

    def run(self):
        self.is_running_now = True
        print("work_is_running")
        for i in range(1, 10):
            time.sleep(1)
            if i % 5 == 0:
                raise Exception(f"invalid_value_of_i_{i}")
            print(f"good_value_of_i_{i}")


class RootFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.simple_work = SimpleWork()

    def run(self):
        print("useless_garbage_log_that_is_always_there_to_overload_logs")
        self.simple_work.run()
        if not self.simple_work.is_running_now:
            pass
            # work is not ready yet
            print("waiting_for_work_to_be_ready")
        else:
            print("flow_and_work_are_running")
            logger.info("logger_flow_work")
            time.sleep(0.1)


if __name__ == "__main__":
    app = LightningApp(RootFlow(), log_level="debug")
