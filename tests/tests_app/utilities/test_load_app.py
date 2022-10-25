import os
from unittest.mock import ANY

import pytest
import tests_app.core.scripts
from tests_app import _PROJECT_ROOT

from lightning_app import LightningApp
from lightning_app.utilities.exceptions import MisconfigurationException
from lightning_app.utilities.load_app import extract_metadata_from_app, load_app_from_file


def test_load_app_from_file():
    app = load_app_from_file(os.path.join(_PROJECT_ROOT, "examples", "app_v0", "app.py"))
    assert isinstance(app, LightningApp)

    test_script_dir = os.path.join(os.path.dirname(tests_app.core.__file__), "scripts")
    with pytest.raises(MisconfigurationException, match="There should not be multiple apps instantiated within a file"):
        load_app_from_file(os.path.join(test_script_dir, "two_apps.py"))

    with pytest.raises(MisconfigurationException, match="The provided file .* does not contain a LightningApp"):
        load_app_from_file(os.path.join(test_script_dir, "empty.py"))

    with pytest.raises(SystemExit, match="1"):
        load_app_from_file(os.path.join(test_script_dir, "script_with_error.py"))


def test_extract_metadata_from_component():
    test_script_dir = os.path.join(os.path.dirname(tests_app.core.__file__), "scripts")
    app = load_app_from_file(os.path.join(test_script_dir, "app_metadata.py"))
    metadata = extract_metadata_from_app(app)
    assert metadata == [
        {
            "affiliation": ["root"],
            "cls_name": "RootFlow",
            "module": "__main__",
            "docstring": "RootFlow.",
        },
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
            "local_build_config": {"__build_config__": ANY},
            "cloud_build_config": {"__build_config__": ANY},
            "cloud_compute": ANY,
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
            "local_build_config": {"__build_config__": ANY},
            "cloud_build_config": {"__build_config__": ANY},
            "cloud_compute": ANY,
        },
        {
            "affiliation": ["root", "flow_b"],
            "cls_name": "FlowB",
            "module": "__main__",
            "docstring": "FlowB.",
        },
        {
            "affiliation": ["root", "flow_b", "work_b"],
            "cls_name": "WorkB",
            "module": "__main__",
            "docstring": "WorkB.",
            "local_build_config": {"__build_config__": ANY},
            "cloud_build_config": {"__build_config__": ANY},
            "cloud_compute": ANY,
        },
    ]
