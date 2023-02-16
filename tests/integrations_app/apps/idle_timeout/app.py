import pathlib

from lightning.app import CloudCompute, LightningApp, LightningFlow, LightningWork
from lightning.app.storage.path import _artifacts_path, _filesystem
from lightning.app.utilities.enum import WorkStageStatus


class SourceFileWriterWork(LightningWork):
    def __init__(self):
        super().__init__(cache_calls=False, parallel=True, cloud_compute=CloudCompute(idle_timeout=5))
        self.counter = 0
        self.value = None
        self.path = None

    def run(self):
        self.path = "lit://boring_file.txt"
        with open(self.path, "w") as f:
            f.write("path")
        self.counter += 1


class DestinationWork(LightningWork):
    def run(self, path):
        assert path.exists()


class RootFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.make_check = True
        self.work = SourceFileWriterWork()
        self.dest_work = DestinationWork(parallel=True)

    def run(self):
        if self.work.counter == 0:
            self.work.run()

        elif self.work.status.stage == WorkStageStatus.STOPPED and self.make_check:
            succeeded_statuses = [status for status in self.work.statuses if status.stage == WorkStageStatus.SUCCEEDED]
            # Ensure the work succeeded at some point
            assert len(succeeded_statuses) > 0
            succeeded_status = succeeded_statuses[-1]

            stopped_status = self.work.statuses[-1]
            assert stopped_status.stage == WorkStageStatus.STOPPED

            # Note: Account for the controlplane, k8s, SIGTERM handler delays.
            assert (stopped_status.timestamp - succeeded_status.timestamp) < 20

            fs = _filesystem()
            destination_path = _artifacts_path(self.work) / pathlib.Path(*self.work.path.resolve().parts[1:])
            assert fs.exists(destination_path)
            self.dest_work.run(self.work.path)
            self.make_check = False
            print("Successfully stopped SourceFileWriterWork.")
        elif self.work.status.stage == WorkStageStatus.STOPPED:
            raise RuntimeError(f"The work has stopped for the wrong reason: {self.work.status.reason}.")

        if self.dest_work.status.stage == WorkStageStatus.SUCCEEDED:
            print("Stopping work")
            self.dest_work.stop()

        if self.dest_work.status.stage == WorkStageStatus.STOPPED:
            print(self.dest_work.statuses)
            print("Application End")
            self.stop()


app = LightningApp(RootFlow(), log_level="debug")
