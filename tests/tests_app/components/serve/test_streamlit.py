import os
import sys
from unittest import mock

import lightning.app
import pytest
from lightning.app.components.serve.streamlit import ServeStreamlit, _build_model, _PatchedWork
from lightning_utilities.core.imports import RequirementCache

_STREAMLIT_AVAILABLE = RequirementCache("streamlit")


class ServeStreamlitTest(ServeStreamlit):
    def __init__(self):
        super().__init__()

        self.test_variable = -1

    @property
    def test_property(self):
        return self.test_variable

    def test_method(self):
        return "test_method"

    @staticmethod
    def test_staticmethod():
        return "test_staticmethod"

    def build_model(self):
        return "model"

    def render():
        pass


@pytest.mark.skipif(not _STREAMLIT_AVAILABLE, reason="requires streamlit")
@mock.patch("lightning.app.components.serve.streamlit.subprocess")
def test_streamlit_start_stop_server(subprocess_mock):
    """Test that `ServeStreamlit.run()` invokes subprocess.Popen with the right parameters."""
    work = ServeStreamlitTest()
    work._name = "test_work"
    work._host = "hostname"
    work._port = 1111

    work.run()

    subprocess_mock.Popen.assert_called_once()

    env_variables = subprocess_mock.method_calls[0].kwargs["env"]
    call_args = subprocess_mock.method_calls[0].args[0]
    assert call_args == [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        lightning.app.components.serve.streamlit.__file__,
        "--server.address",
        "hostname",
        "--server.port",
        "1111",
        "--server.headless",
        "true",
    ]

    assert env_variables["LIGHTNING_COMPONENT_NAME"] == "test_work"
    assert env_variables["LIGHTNING_WORK"] == "ServeStreamlitTest"
    assert env_variables["LIGHTNING_WORK_MODULE_FILE"] == __file__

    assert "LIGHTNING_COMPONENT_NAME" not in os.environ
    assert "LIGHTNING_WORK" not in os.environ
    assert "LIGHTNING_WORK_MODULE_FILE" not in os.environ

    work.on_exit()
    subprocess_mock.Popen().kill.assert_called_once()


def test_patched_work():
    class TestState:
        test_variable = 1

    patched_work = _PatchedWork(TestState(), ServeStreamlitTest)

    assert patched_work.test_variable == 1
    assert patched_work.test_property == 1
    assert patched_work.test_method() == "test_method"
    assert patched_work.test_staticmethod() == "test_staticmethod"


@pytest.mark.skipif(not _STREAMLIT_AVAILABLE, reason="requires streamlit")
def test_build_model():
    import streamlit as st

    st.session_state = {}
    st.spinner = mock.MagicMock()

    class TestState:
        test_variable = 1

    patched_work = _PatchedWork(TestState(), ServeStreamlitTest)
    patched_work.build_model = mock.MagicMock(return_value="test_model")

    _build_model(patched_work)

    assert st.session_state["_model"] == "test_model"
    assert patched_work.model == "test_model"
    patched_work.build_model.assert_called_once()

    patched_work.build_model.reset_mock()

    _build_model(patched_work)

    patched_work.build_model.assert_not_called()
