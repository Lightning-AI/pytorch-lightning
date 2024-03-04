import os
import pathlib
from copy import deepcopy
from time import sleep

import pytest
from deepdiff import DeepDiff
from lightning.app import LightningFlow, LightningWork
from lightning.app.core.app import LightningApp
from lightning.app.runners import MultiProcessRuntime
from lightning.app.storage.drive import Drive, _maybe_create_drive
from lightning.app.utilities.component import _set_flow_context


class SyncWorkLITDriveA(LightningWork):
    def __init__(self, tmpdir):
        super().__init__()
        self.tmpdir = tmpdir

    def run(self, drive: Drive):
        with open(f"{self.tmpdir}/a.txt", "w") as f:
            f.write("example")

        drive.root_folder = self.tmpdir
        drive.put("a.txt")
        os.remove(f"{self.tmpdir}/a.txt")


class SyncWorkLITDriveB(LightningWork):
    def run(self, drive: Drive):
        assert not os.path.exists("a.txt")
        drive.get("a.txt")
        assert os.path.exists("a.txt")


class SyncFlowLITDrives(LightningFlow):
    def __init__(self, tmpdir):
        super().__init__()
        self.log_dir = Drive("lit://log_dir")
        self.work_a = SyncWorkLITDriveA(str(tmpdir))
        self.work_b = SyncWorkLITDriveB()

    def run(self):
        self.work_a.run(self.log_dir)
        self.work_b.run(self.log_dir)
        self.stop()


@pytest.mark.flaky(reruns=3, reruns_delay=5)  # todo: likely dead feature, fine to crash...
def test_synchronization_lit_drive(tmpdir):
    if os.path.exists("a.txt"):
        os.remove("a.txt")
    app = LightningApp(SyncFlowLITDrives(tmpdir))
    MultiProcessRuntime(app, start_server=False).dispatch()
    if os.path.exists("a.txt"):
        os.remove("a.txt")


class LITDriveWork(LightningWork):
    def __init__(self):
        super().__init__(parallel=True)
        self.drive = None
        self.counter = 0

    def run(self, *args, **kwargs):
        if self.counter == 0:
            self.drive = Drive("lit://this_drive_id")
            sleep(10)
            with open("a.txt", "w") as f:
                f.write("example")

            self.drive.put("a.txt")
        else:
            assert self.drive
            assert self.drive.list(".") == ["a.txt"]
            self.drive.delete("a.txt")
            assert self.drive.list(".") == []
        self.counter += 1


class LITDriveWork2(LightningWork):
    def __init__(self):
        super().__init__(parallel=True)

    def run(self, drive: Drive, **kwargs):
        assert drive.list(".") == []
        drive.get("a.txt", timeout=60)
        assert drive.list(".") == ["a.txt"]
        assert drive.list(".", component_name=self.name) == []


class LITDriveFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.work = LITDriveWork()
        self.work2 = LITDriveWork2()

    def run(self):
        self.work.run("0")
        if self.work.drive:
            self.work2.run(self.work.drive, something="hello")
        if self.work2.has_succeeded:
            self.work.run("1")
        if self.work.counter == 2:
            self.stop()


@pytest.mark.flaky(reruns=3, reruns_delay=5)  # todo: likely dead feature, fine to crash...
def test_lit_drive_transferring_files():
    app = LightningApp(LITDriveFlow())
    MultiProcessRuntime(app, start_server=False).dispatch()
    os.remove("a.txt")


@pytest.mark.xfail(strict=False)  # todo: likely dead feature, fine to crash...
def test_lit_drive():
    with pytest.raises(Exception, match="Unknown protocol for the drive 'id' argument"):
        Drive("invalid_drive_id")

    with pytest.raises(
        Exception, match="The id should be unique to identify your drive. Found `this_drive_id/something_else`."
    ):
        Drive("lit://this_drive_id/something_else")

    drive = Drive("lit://this_drive_id")
    with pytest.raises(Exception, match="The component name needs to be known to put a path to the Drive."):
        drive.put(".")

    with pytest.raises(Exception, match="The component name needs to be known to delete a path to the Drive."):
        drive.delete(".")

    with open("a.txt", "w") as f:
        f.write("example")

    os.makedirs("checkpoints")
    with open("checkpoints/a.txt", "w") as f:
        f.write("example")

    drive = Drive("lit://drive_1", allow_duplicates=False)
    drive.component_name = "root.work_1"
    assert drive.list(".") == []
    drive.put("a.txt")
    assert drive.list(".") == ["a.txt"]
    drive.component_name = "root.work_2"
    with pytest.raises(Exception, match="The file a.txt can't be added as already found in the Drive."):
        drive.put("a.txt")
    drive.get("a.txt")

    drive = Drive("lit://drive_2", allow_duplicates=False)
    drive.component_name = "root.work_1"
    drive.put("checkpoints/a.txt")
    drive.component_name = "root.work_2"
    with pytest.raises(Exception, match="The file checkpoints/a.txt can't be added as already found in the Drive."):
        drive.put("checkpoints/a.txt")

    drive = Drive("lit://drive_3", allow_duplicates=False)
    drive.component_name = "root.work_1"
    drive.put("checkpoints/")
    drive.component_name = "root.work_2"
    with pytest.raises(Exception, match="The file checkpoints/a.txt can't be added as already found in the Drive."):
        drive.put("checkpoints/a.txt")

    drive = Drive("lit://drive_3", allow_duplicates=True)
    drive.component_name = "root.work_1"
    drive.put("checkpoints/")
    drive.component_name = "root.work_2"
    with pytest.raises(
        Exception, match="The file checkpoints/a.txt doesn't exists in the component_name space root.work_2."
    ):
        drive.delete("checkpoints/a.txt")
    drive.put("checkpoints/a.txt")
    drive.delete("checkpoints/a.txt")

    drive = Drive("lit://drive_3", allow_duplicates=True)
    drive.component_name = "root.work_1"
    drive.put("checkpoints/")
    with pytest.raises(Exception, match="['root.work_1', 'root.work_2']"):
        drive.get("checkpoints/")
    drive.get("checkpoints/a.txt", component_name="root.work_1")
    drive.get("checkpoints/a.txt", component_name="root.work_1", timeout=1)

    with pytest.raises(FileNotFoundError):
        drive.get("checkpoints/b.txt", component_name="root.work_1")
    with pytest.raises(Exception, match="The following checkpoints/b.txt wasn't found in 1 seconds"):
        drive.get("checkpoints/b.txt", component_name="root.work_1", timeout=1)
    drive.component_name = "root.work_2"
    drive.put("checkpoints/")
    drive.component_name = "root.work_3"
    with pytest.raises(Exception, match="We found several matching files created by multiples components"):
        drive.get("checkpoints/a.txt")
    with pytest.raises(Exception, match="We found several matching files created by multiples components"):
        drive.get("checkpoints/a.txt", timeout=1)

    drive = Drive("lit://drive_4", allow_duplicates=True)
    drive.component_name = "root.work_1"
    with pytest.raises(Exception, match="The following checkpoints/a.txt wasn't found in 1 seconds."):
        drive.get("checkpoints/a.txt", timeout=1)

    drive = Drive("lit://test", allow_duplicates=True)
    drive.component_name = "root.work1"
    drive.put("checkpoints")
    drive.get("checkpoints", overwrite=True)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        drive.get("checkpoints")

    drive = Drive("lit://drive_5", allow_duplicates=True)
    drive.component_name = "root.work"
    _set_flow_context()
    with pytest.raises(Exception, match="The flow isn't allowed to put files into a Drive."):
        drive.put("a.txt")
    with pytest.raises(Exception, match="The flow isn't allowed to list files from a Drive."):
        drive.list("a.txt")
    with pytest.raises(Exception, match="The flow isn't allowed to get files from a Drive."):
        drive.get("a.txt")

    os.remove("checkpoints/a.txt")
    os.rmdir("checkpoints")
    os.remove("a.txt")


@pytest.mark.parametrize("drive_id", ["lit://drive"])
def test_maybe_create_drive(drive_id):
    drive = Drive(drive_id, allow_duplicates=False)
    drive.component_name = "root.work1"
    assert isinstance(drive.root_folder, pathlib.Path)
    drive_state = drive.to_dict()
    assert isinstance(drive_state["root_folder"], str)
    new_drive = _maybe_create_drive(drive.component_name, drive.to_dict())
    assert isinstance(drive.root_folder, pathlib.Path)
    assert new_drive.protocol == drive.protocol
    assert new_drive.id == drive.id
    assert new_drive.component_name == drive.component_name
    drive_state["root_folder"] = pathlib.Path(drive_state["root_folder"])
    copy_drive_state = deepcopy(drive_state)
    deep_diff = DeepDiff(copy_drive_state, drive_state)
    assert "unprocessed" in deep_diff
    deep_diff.pop("unprocessed")


@pytest.mark.parametrize("drive_id", ["lit://drive"])
def test_drive_deepcopy(drive_id):
    drive = Drive(drive_id, allow_duplicates=True)
    drive.component_name = "root.work1"
    new_drive = deepcopy(drive)
    assert new_drive.id == drive.id
    assert new_drive.component_name == drive.component_name


def test_s3_drive_raises_error_telling_users_to_use_mounts():
    with pytest.raises(ValueError, match="Using S3 buckets in a Drive is no longer supported."):
        Drive("s3://foo/")


def test_drive_root_folder_breaks():
    with pytest.raises(Exception, match="The provided root_folder isn't a directory: a"):
        Drive("lit://drive", root_folder="a")
