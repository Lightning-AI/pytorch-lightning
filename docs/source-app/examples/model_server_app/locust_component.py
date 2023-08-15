import os
import subprocess

from lightning import BuildConfig, LightningWork


class Locust(LightningWork):
    def __init__(self, num_users: int = 100):
        """This component checks the performance of a server. The server url is passed to its run method.

        Arguments:
            num_users: Number of users emulated by Locust

        """
        # Note: Using the default port 8089 of Locust.
        super().__init__(
            port=8089,
            parallel=True,
            cloud_build_config=BuildConfig(requirements=["locust"]),
        )
        self.num_users = num_users

    def run(self, load_tested_url: str):
        # 1: Create the locust command line.
        cmd = " ".join(
            [
                "locust",
                "--master-host",
                str(self.host),
                "--master-port",
                str(self.port),
                "--host",
                str(load_tested_url),
                "-u",
                str(self.num_users),
            ]
        )
        # 2: Create another process with locust
        process = subprocess.Popen(cmd, cwd=os.path.dirname(__file__), shell=True)

        # 3: Wait for the process to finish. As locust is a server,
        # this waits infinitely or if killed.
        process.wait()
