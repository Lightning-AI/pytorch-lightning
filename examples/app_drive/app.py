import os

import lightning as L
from lightning.app.storage.drive import Drive


class Work_1(L.LightningWork):
    def run(self, drive: Drive):
        # 1. Create a file.
        with open("a.txt", "w") as f:
            f.write("Hello World !")

        # 2. Put the file into the drive.
        drive.put("a.txt")

        # 3. Delete the locally.
        os.remove("a.txt")


class Work_2(L.LightningWork):
    def __init__(self):
        super().__init__()

    def run(self, drive: Drive):
        print(drive.list("."))  # Prints ["a.txt"]

        print(os.path.exists("a.txt"))  # Prints False

        drive.get("a.txt")  # Transfer the file from this drive to the local filesystem.

        print(os.path.exists("a.txt"))  # Prints True

        with open("a.txt") as f:
            print(f.readlines()[0])  # Prints Hello World !


class Flow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.drive_1 = Drive("lit://drive_1")
        self.work_1 = Work_1()
        self.work_2 = Work_2()

    def run(self):
        # Pass the drive to both works.
        self.work_1.run(self.drive_1)
        self.work_2.run(self.drive_1)
        self._exit("Application End!")


app = L.LightningApp(Flow(), debug=True)
