import os
import sys
from unittest.mock import ANY

import pytest
from lightning.app.utilities.exceptions import MisconfigurationException
from lightning.app.utilities.load_app import extract_metadata_from_app, load_app_from_file


def test_load_app_from_file_errors():
    test_script_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "core", "scripts")
    with pytest.raises(MisconfigurationException, match="There should not be multiple apps instantiated within a file"):
        load_app_from_file(os.path.join(test_script_dir, "two_apps.py"))

    with pytest.raises(MisconfigurationException, match="The provided file .* does not contain a LightningApp"):
        load_app_from_file(os.path.join(test_script_dir, "empty.py"))

    with pytest.raises(SystemExit, match="1"):
        load_app_from_file(os.path.join(test_script_dir, "script_with_error.py"))


@pytest.mark.parametrize("app_path", ["app_metadata.py", "app_with_local_import.py"])
def test_load_app_from_file(app_path):
    """Test that apps load without error and that sys.path and main module are set."""
    original_main = sys.modules["__main__"]
    test_script_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "core", "scripts")
    load_app_from_file(os.path.join(test_script_dir, app_path), raise_exception=True)

    assert test_script_dir in sys.path
    assert sys.modules["__main__"] != original_main


def test_extract_metadata_from_component():
    test_script_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "core", "scripts")
    app = load_app_from_file(os.path.join(test_script_dir, "app_metadata.py"))
    metadata = extract_metadata_from_app(app)
    assert metadata == [
        {"affiliation": ["root"], "cls_name": "RootFlow", "module": "__main__", "docstring": "RootFlow."},
        {
            "affiliation": ["root", "flow_a_1"],
            "cls_name": "FlowA",
            "module": "__main__",
            "docstring": "FlowA Component.",
        },
        {
            "affiliation": ["root", "flow_a_1", "work_a"],
            "cls_name": "WorkA",
            "module": "__main__",
            "docstring": "WorkA.",
            "local_build_config": {"__build_config__": {"requirements": [], "dockerfile": None, "image": None}},
            "cloud_build_config": {"__build_config__": {"requirements": [], "dockerfile": None, "image": None}},
            "cloud_compute": {
                "type": "__cloud_compute__",
                "name": "cpu-small",
                "disk_size": 0,
                "idle_timeout": None,
                "shm_size": 0,
                "mounts": None,
                "_internal_id": "default",
                "interruptible": False,
                "colocation_group_id": None,
            },
        },
        {
            "affiliation": ["root", "flow_a_2"],
            "cls_name": "FlowA",
            "module": "__main__",
            "docstring": "FlowA Component.",
        },
        {
            "affiliation": ["root", "flow_a_2", "work_a"],
            "cls_name": "WorkA",
            "module": "__main__",
            "docstring": "WorkA.",
            "local_build_config": {"__build_config__": {"requirements": [], "dockerfile": None, "image": None}},
            "cloud_build_config": {"__build_config__": {"requirements": [], "dockerfile": None, "image": None}},
            "cloud_compute": {
                "type": "__cloud_compute__",
                "name": "cpu-small",
                "disk_size": 0,
                "idle_timeout": None,
                "shm_size": 0,
                "mounts": None,
                "_internal_id": "default",
                "interruptible": False,
                "colocation_group_id": None,
            },
        },
        {"affiliation": ["root", "flow_b"], "cls_name": "FlowB", "module": "__main__", "docstring": "FlowB."},
        {
            "affiliation": ["root", "flow_b", "work_b"],
            "cls_name": "WorkB",
            "module": "__main__",
            "docstring": "WorkB.",
            "local_build_config": {"__build_config__": {"requirements": [], "dockerfile": None, "image": None}},
            "cloud_build_config": {"__build_config__": {"requirements": [], "dockerfile": None, "image": None}},
            "cloud_compute": {
                "type": "__cloud_compute__",
                "name": "gpu",
                "disk_size": 0,
                "idle_timeout": None,
                "shm_size": 1024,
                "mounts": None,
                "_internal_id": ANY,
                "interruptible": False,
                "colocation_group_id": None,
            },
        },
    ]
