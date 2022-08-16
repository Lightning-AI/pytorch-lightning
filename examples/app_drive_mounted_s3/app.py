"""Demonstrates how a Drive can be used to mount an AWS s3 bucket to a Work's filesystem.

This can be useful for loading larges datasets into a Work while training
machine learning models or doing other compute / data heavy tasks.

Notes:
  - At the moment, only public S3 buckets are supported.
  - A mount is only created when executing the app on the cloud. When executing
    in the local runtime, the `root_folder` (mount path for the bucket) should
    contain a sample copy of the data / folder structure of the s3 bucket. This
    will allow you to simulate building an app running on the full dataset locally
    without having to worry about setting up complex tooling or shims to put the
    data on your local system.
"""
import glob
import os
import time

import lightning as L
from lightning.app.storage.drive import Drive

MINUTE = 60  # 60 seconds in a min


class Work1(L.LightningWork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # signal to indicate if work has completed executing or not
        self.done = False

        # Note: each work runs independently. This mount (root_folder) is the same path
        #       which is used in the drive passed to Work2, but it may not necessarily
        #       contain the same contents unless the same s3 source bucket is specified.
        #       This only applies when running on the cloud. When on your local machine,
        #       we pass through to whatever is defined on your filesystem at this path.
        self.drive_1_mountdir = "/tmp/drive1/datadir/"
        self.drive_2_mountdir = "/tmp/drive2/datadir/"

        # If these directories do not exist on your local machine, be sure to create them
        # because the work initialization runs once on your machine the first time the
        # app is sent to the cloud.
        os.makedirs(self.drive_1_mountdir, exist_ok=True)
        os.makedirs(self.drive_2_mountdir, exist_ok=True)

        # Note: When initializing s3 drives in a Work (rather than a flow) please keep in mind that
        #       they must be initialized within the __init__ body. When running on the cloud, an
        #       s3 drive cannot be dynamically mounted in a Work within the `.run()` method.
        self.drive_1 = Drive("s3://ryft-public-sample-data/esRedditJson/", root_folder=self.drive_1_mountdir)
        self.drive_2 = Drive("s3://ryft-public-sample-data/", root_folder=self.drive_2_mountdir)

        # files which i'm going to open below just to check that the bucket contents
        # are actually mounted
        self.drive_1_checkfile = os.path.join(self.drive_1_mountdir, "esRedditJson11")
        self.drive_2_checkfile = os.path.join(self.drive_2_mountdir, "twitter", "twitter2015.tar.gz")

    def run(self):
        # -------- Check the first drive attached to this work --------

        drive_1_filenames = set()
        for filename in glob.iglob(f"{self.drive_1_mountdir}**/**", recursive=True):
            print(filename)
            drive_1_filenames.add(filename)

        # ensure that the file exists in the printed list
        assert self.drive_1_checkfile in drive_1_filenames
        # assert that content size is reported and not just inode metadata size
        assert os.path.getsize(self.drive_1_checkfile) > 1_000  # bytes
        # verify that we can open and read the contents without errors
        with open(self.drive_1_checkfile) as f:
            f.read()

        print(f"Drive1(id={self.drive_1.id}) was mounted successfully!")

        # -------- Now check the second drive in this work --------

        drive_2_filenames = set()
        for filename in glob.iglob(f"{self.drive_2_mountdir}**/**", recursive=True):
            print(filename)
            drive_2_filenames.add(filename)

        # ensure that the file exists in the printed list
        assert self.drive_2_checkfile in drive_2_filenames
        # assert that content size is reported and not just inode metadata size
        assert os.path.getsize(self.drive_2_checkfile) > 1_000  # bytes
        # verify that we can open and read the contents without errors
        with open(self.drive_2_checkfile, "rb") as f:
            f.read()

        print(f"Drive2(id={self.drive_2.id}) was mounted successfully!")

        # -------- Done! --------
        self.done = True


class Work2(L.LightningWork):
    def __init__(self, drive: "Drive", *args, **kwargs):
        super().__init__(*args, **kwargs)

        # you must assign the drive to an attribute on the work in the __init__ method
        # in order for it to be mounted when running on the cloud runtime.
        self.drive = drive

        # signal to indicate if work has completed executing or not
        self.done = None

    def run(self, check_file: str):
        print(f"verifying that {check_file} exists")
        assert os.path.exists(check_file)

        print(f"verifying that the byte size of {check_file} is reasonably reported & not just the inode metadata")
        assert os.path.getsize(check_file) > 1_000  # bytes

        print(f"verifying that we can open and read the contents of {check_file} without errors")
        with open(check_file, "rb") as f:
            f.read()

        print("verifications complete! Exiting work run...")
        self.done = True


class Flow(L.LightningFlow):
    def __init__(self):
        super().__init__()

        # Work1 contains Drive definitions in the __init__() method
        self.work_1 = Work1(parallel=True)

        # Work 2 will pass a drive we define in the flow into the work as an init argument
        self.work2_drive_root_folder = "/tmp/drive1/datadir/"
        # The directory must exist when initializing the flow and work, even if it is empty.
        # We will create, and mount to this path automatically when `Work.run()` is called.
        os.makedirs(self.work2_drive_root_folder, exist_ok=True)

        # create the drive and initialize Work2 with the drive object
        drive = Drive("s3://ryft-public-sample-data/twitter/", root_folder=self.work2_drive_root_folder)
        self.work_2 = Work2(drive, parallel=True)

        # we will call Work2.run() with a file path to check to ensure that it is available and mounted.
        self.work2_drive_checkfile = os.path.join(self.work2_drive_root_folder, "twitter2015.tar.gz")

        self.start_time = None

    def run(self):
        # set the start time only once for the Flow loop
        if self.start_time is None:
            self.start_time = time.time()

        # Works which pass through the `parallel` argument to the L.Work base class run only once
        # even though the main flow loop repeats infinitely if not explicitly exited.
        self.work_1.run()
        self.work_2.run(self.work2_drive_checkfile)

        # are we done? If so, exit!
        if self.work_1.done and self.work_2.done:
            self._exit("Application End Successful!")

        # If the works don't complete in < 30 minutes, a problem occured and we will exit here.
        if time.time() - self.start_time > 30 * MINUTE:
            raise TimeoutError("More than 30 minutes elapsed since starting the flow")

        # sleep before the next Flow loop.
        time.sleep(0.1)


app = L.LightningApp(Flow())
