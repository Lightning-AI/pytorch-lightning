import pathlib

from lightning_app import CloudCompute, LightningApp, LightningFlow, LightningWork
from lightning_app.storage.path import artifacts_path, filesystem
from lightning_app.utilities.enum import WorkStageStatus, WorkStopReasons


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

        elif (
            self.work.status.stage == WorkStageStatus.STOPPED
            and self.work.status.reason == WorkStopReasons.SIGTERM_SIGNAL_HANDLER
            and self.make_check
        ):
            succeeded_status = self.work.statuses[-3]
            stopped_status_pending = self.work.statuses[-2]
            stopped_status_sigterm = self.work.statuses[-1]
            assert succeeded_status.stage == WorkStageStatus.SUCCEEDED
            assert stopped_status_pending.stage == WorkStageStatus.STOPPED
            assert stopped_status_pending.reason == WorkStopReasons.PENDING
            assert stopped_status_sigterm.stage == WorkStageStatus.STOPPED
            assert stopped_status_sigterm.reason == WorkStopReasons.SIGTERM_SIGNAL_HANDLER
            # Note: Account for the controlplane, k8s, SIGTERM handler delays.
            assert (stopped_status_pending.timestamp - succeeded_status.timestamp) < 20
            assert (stopped_status_sigterm.timestamp - stopped_status_pending.timestamp) < 120
            fs = filesystem()
            destination_path = artifacts_path(self.work) / pathlib.Path(*self.work.path.resolve().parts[1:])
            assert fs.exists(destination_path)
            self.dest_work.run(self.work.path)
            self.make_check = False
            print("Successfully stopped SourceFileWriterWork.")

        if self.dest_work.status.stage == WorkStageStatus.SUCCEEDED:
            print("Stopping work")
            self.dest_work.stop()

        if self.dest_work.status.stage == WorkStageStatus.STOPPED:
            print(self.dest_work.statuses)
            print("Application End")
            self._exit()


app = LightningApp(RootFlow(), debug=True)
