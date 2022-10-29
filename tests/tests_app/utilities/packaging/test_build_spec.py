import os
import sys

from tests_app import _TESTS_ROOT

from lightning_app.testing import application_testing, LightningTestApp
from lightning_app.utilities.packaging.build_config import BuildConfig

EXTRAS_ARGS = ["--blocking", "False", "--multiprocess", "--open-ui", "False"]


class NoRequirementsLightningTestApp(LightningTestApp):
    def on_after_run_once(self):
        assert self.root.work.local_build_config.requirements == []
        assert self.root.work.cloud_build_config.requirements == []
        return super().on_after_run_once()


def test_build_config_no_requirements():
    command_line = [os.path.join(_TESTS_ROOT, "utilities/packaging/projects/no_req/app.py")]
    application_testing(NoRequirementsLightningTestApp, command_line + EXTRAS_ARGS)
    sys.path = sys.path[:-1]


def test_build_config_requirements_provided():
    spec = BuildConfig(requirements=["dask", "./projects/req/comp_req/a/requirements.txt"])
    assert spec.requirements == [
        "dask",
        "pandas",
        "pytorch_" + "lightning==1.5.9",  # ugly hack due to replacing `pytorch_lightning string`
        "git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0",
    ]
    assert spec == BuildConfig.from_dict(spec.to_dict())


class BuildSpecTest(BuildConfig):
    def build_commands(self):
        return super().build_commands() + ["pip install redis"]


def test_build_config_invalid_requirements():
    spec = BuildSpecTest(requirements=["./projects/requirements.txt"])
    assert spec.requirements == ["cloud-stars"]
    assert spec.build_commands() == ["pip install redis"]


def test_build_config_dockerfile_provided():
    spec = BuildConfig(dockerfile="./projects/Dockerfile.cpu")
    assert not spec.requirements
    # ugly hack due to replacing `pytorch_lightning string
    assert "pytorchlightning/pytorch_" + "lightning" in spec.dockerfile[0]


class DockerfileLightningTestApp(LightningTestApp):
    def on_after_run_once(self):
        print(self.root.work.local_build_config.dockerfile)
        # ugly hack due to replacing `pytorch_lightning string
        assert "pytorchlightning/pytorch_" + "lightning" in self.root.work.local_build_config.dockerfile[0]
        return super().on_after_run_once()


def test_build_config_dockerfile():
    command_line = [os.path.join(_TESTS_ROOT, "utilities/packaging/projects/dockerfile/app.py")]
    application_testing(DockerfileLightningTestApp, command_line + EXTRAS_ARGS)
    sys.path = sys.path[:-1]


class RequirementsLightningTestApp(LightningTestApp):
    def on_after_run_once(self):
        assert self.root.work.local_build_config.requirements == [
            "git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0",
            "pandas",
            "pytorch_" + "lightning==1.5.9",  # ugly hack due to replacing `pytorch_lightning string
        ]
        return super().on_after_run_once()


def test_build_config_requirements():
    command_line = [os.path.join(_TESTS_ROOT, "utilities/packaging/projects/req/app.py")]
    application_testing(RequirementsLightningTestApp, command_line + EXTRAS_ARGS)
    sys.path = sys.path[:-1]
