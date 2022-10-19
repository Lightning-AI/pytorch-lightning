import json
import os
from unittest import mock

import pytest
from tests_app import _PROJECT_ROOT

from lightning_app import LightningApp, LightningFlow, LightningWork  # F401


class SavedAppExample(LightningFlow):
    def __init__(self):
        super().__init__()
        self.text = ""

    def run(self):
        print(f"what is the value of text? {self.text}")


class DynamicWorkLevel3(LightningWork):
    def __init__(self):
        super().__init__()
        self.text = ""

    def run(self):
        print(f"what is the value of text in the work? {self.text}")


class DynamicFlowLevel2(LightningFlow):
    def __init__(self):
        super().__init__()
        self.text = ""

    def run(self):
        if not getattr(self, "work_level_3", None):
            self.work_level_3 = DynamicWorkLevel3()
        self.work_level_3.run()

    def load_state_dict(self, flow_state, children_states, strict) -> None:
        self.work_level_3 = DynamicWorkLevel3()
        super().load_state_dict(flow_state, children_states, strict=strict)


class DynamicFlowLevel1(LightningFlow):
    def __init__(self):
        super().__init__()
        self.text = ""

    def run(self):
        if not getattr(self, "flow_level_2", None):
            self.flow_level_2 = DynamicFlowLevel2()
        self.flow_level_2.run()

    def load_state_dict(self, flow_state, children_states, strict) -> None:
        self.flow_level_2 = DynamicFlowLevel2()
        super().load_state_dict(flow_state, children_states, strict=strict)


class SavedDynamicAppExample(LightningFlow):
    def __init__(self):
        super().__init__()
        self.text = ""

    def run(self):
        if not getattr(self, "flow_level_1", None):
            self.flow_level_1 = DynamicFlowLevel1()
        self.flow_level_1.run()

    def load_state_dict(self, flow_state, children_states, strict) -> None:
        self.flow_level_1 = DynamicFlowLevel1()
        super().load_state_dict(flow_state, children_states, strict=strict)


class SavedDynamicAppExampleInvalid(LightningFlow):
    def __init__(self):
        super().__init__()
        self.text = ""

    def run(self):
        if not getattr(self, "flow_level_1", None):
            self.flow_level_1 = DynamicFlowLevel1()
        self.flow_level_1.run()


def test_load_app_from_local_checkpoint():
    app = LightningApp(SavedAppExample())
    app.load_app_state_from_checkpoint(
        os.path.join(_PROJECT_ROOT, "tests/tests_app/test_date/saved_app_checkpoint.json")
    )
    assert app.root.text == "Hello World!"


def test_load_dynamic_app_from_local_checkpoint():
    app = LightningApp(SavedDynamicAppExample())

    # This is the checkpoint file that was saved from the SavedDynamicAppExample app and contains the state of the
    # dynamic components flow and work and the state contains updated value of the text attribute of each component.
    app.load_app_state_from_checkpoint(
        os.path.join(_PROJECT_ROOT, "tests/tests_app/test_date/saved_dynamic_app_checkpoint.json")
    )
    assert app.root.text == "Hello Root Flow !"
    assert app.root.flow_level_1.text == "Hello Flow Level 1!"
    assert app.root.flow_level_1.flow_level_2.text == "Hello Flow Level 2!"
    assert app.root.flow_level_1.flow_level_2.work_level_3.text == "Hello Work Level 3!"


def test_load_dynamic_app_from_local_checkpoint_invalid_app():
    # Here we are missing the load_state_dict method in SavedDynamicAppExampleInvalid that loads the dynamic components
    # and does state migration if needed. see LightningFlow.load_state_dict() for more details.
    app = LightningApp(SavedDynamicAppExampleInvalid())

    with pytest.raises(ValueError, match="The component flow_level_1 wasn't instantiated for the component root"):
        app.load_app_state_from_checkpoint(
            os.path.join(_PROJECT_ROOT, "tests/tests_app/test_date/saved_dynamic_app_checkpoint.json")
        )


@pytest.mark.parametrize(
    "input_checkpoint, expected_selected_checkpoint",
    [
        ["latest", "lightningapp_checkpoint_1665501576.json"],
        ["lightningapp_checkpoint_1665501575.json", "lightningapp_checkpoint_1665501575.json"],
    ],
)
@mock.patch("lightning_app.core.app.Drive")
@mock.patch("lightning_app.core.app.LightningApp._load_checkpoint_from_json_file")
@mock.patch("os.path.exists", mock.MagicMock(return_value=True))
def test_load_app_from_checkpoint_on_drive(
    load_checkpoint_mock, drive_mock, input_checkpoint, expected_selected_checkpoint
):

    mocked_files_on_drive = [
        "lightningapp_checkpoint_1665501574.json",
        "lightningapp_checkpoint_1665501575.json",
        "lightningapp_checkpoint_1665501576.json",  # latest
    ]

    drive_mock.return_value.list.return_value = mocked_files_on_drive
    load_checkpoint_mock.return_value = json.load(
        open(os.path.join(_PROJECT_ROOT, "tests/tests_app/test_date/saved_app_checkpoint.json"))
    )
    app = LightningApp(SavedAppExample())
    state = app._get_checkpoint_if_available_on_drive(input_checkpoint)
    app.load_state_dict(state)
    assert app.root.text == "Hello World!"
    load_checkpoint_mock.assert_called_once_with(expected_selected_checkpoint)
