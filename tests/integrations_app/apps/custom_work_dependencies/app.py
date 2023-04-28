import os

from lightning.app import BuildConfig, CloudCompute, LightningApp, LightningFlow, LightningWork


class CustomBuildConfig(BuildConfig):
    def build_commands(self):
        return ["sudo apt update", "sudo apt install redis", "pip install lmdb"]


class WorkWithCustomDeps(LightningWork):
    def __init__(self, cloud_compute: CloudCompute = CloudCompute(), **kwargs):
        build_config = CustomBuildConfig(requirements=["py"])
        super().__init__(parallel=True, **kwargs, cloud_compute=cloud_compute, cloud_build_config=build_config)

    def run(self):
        # installed by the build commands and by requirements in the build config
        import lmdb

        print("installed lmdb version:", lmdb.__version__)


class WorkWithCustomBaseImage(LightningWork):
    def __init__(self, cloud_compute: CloudCompute = CloudCompute(), **kwargs):
        # this image has been created from ghcr.io/gridai/base-images:v1.8-cpu
        # by just adding an empty file at /content/.e2e_test
        image_tag = os.getenv("LIGHTNING_E2E_TEST_IMAGE_VERSION", "v1.29")
        custom_image = f"ghcr.io/gridai/image-for-testing-custom-images-in-e2e:{image_tag}"
        build_config = BuildConfig(image=custom_image)
        super().__init__(parallel=True, **kwargs, cloud_compute=cloud_compute, cloud_build_config=build_config)

    def run(self):
        # checking the existence of the file - this file had been added to the custom base image
        assert ".e2e_test" in os.listdir("/testdir/"), "file not found"


class CustomWorkBuildConfigChecker(LightningFlow):
    def run(self):
        # create dynamically the work at runtime
        if not hasattr(self, "work1"):
            self.work1 = WorkWithCustomDeps()
        if not hasattr(self, "work2"):
            self.work2 = WorkWithCustomBaseImage()

        self.work1.run()
        self.work2.run()

        if self.work1.has_succeeded and self.work2.has_succeeded:
            print("--- Custom Work Dependency checker End ----")
            self.stop()


app = LightningApp(CustomWorkBuildConfigChecker())
