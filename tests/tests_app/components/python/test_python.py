import os
import tarfile

import pytest
from lightning.app.components.python import PopenPythonScript, TracerPythonScript
from lightning.app.components.python.tracer import Code
from lightning.app.storage.drive import Drive
from lightning.app.testing.helpers import _RunIf
from lightning.app.testing.testing import run_work_isolated
from lightning.app.utilities.component import _set_work_context
from lightning.app.utilities.enum import CacheCallsKeys
from tests_app import _PROJECT_ROOT

COMPONENTS_SCRIPTS_FOLDER = str(os.path.join(_PROJECT_ROOT, "tests/tests_app/components/python/scripts/"))


def test_non_existing_python_script():
    match = "tests/components/python/scripts/0.py"
    with pytest.raises(FileNotFoundError, match=match):
        python_script = PopenPythonScript(match)
        run_work_isolated(python_script)
        assert not python_script.has_started

    python_script = TracerPythonScript(match, raise_exception=False)
    run_work_isolated(python_script)
    assert python_script.has_failed


def test_simple_python_script():
    python_script = PopenPythonScript(COMPONENTS_SCRIPTS_FOLDER + "a.py")
    run_work_isolated(python_script)
    assert python_script.has_succeeded

    python_script = TracerPythonScript(COMPONENTS_SCRIPTS_FOLDER + "a.py")
    run_work_isolated(python_script)
    assert python_script.has_succeeded


def test_simple_popen_python_script_with_kwargs():
    python_script = PopenPythonScript(
        COMPONENTS_SCRIPTS_FOLDER + "b.py",
        script_args="--arg_0=hello --arg_1=world",
    )
    run_work_isolated(python_script)
    assert python_script.has_succeeded


@_RunIf(skip_windows=True)
def test_popen_python_script_failure():
    python_script = PopenPythonScript(
        COMPONENTS_SCRIPTS_FOLDER + "c.py",
        env={"VARIABLE": "1"},
        raise_exception=False,
    )
    run_work_isolated(python_script)
    assert python_script.has_failed
    assert "Exception(self.exit_code)" in python_script.status.message


def test_tracer_python_script_with_kwargs():
    python_script = TracerPythonScript(
        COMPONENTS_SCRIPTS_FOLDER + "b.py",
        script_args="--arg_0=hello --arg_1=world",
        raise_exception=False,
    )
    run_work_isolated(python_script)
    assert python_script.has_succeeded

    python_script = TracerPythonScript(
        COMPONENTS_SCRIPTS_FOLDER + "c.py",
        env={"VARIABLE": "1"},
        raise_exception=False,
    )
    run_work_isolated(python_script)
    assert python_script.has_failed


def test_tracer_component_with_code():
    """This test ensures the Tracer Component gets the latest code from the code object that is provided and arguments
    are cleaned."""

    drive = Drive("lit://code")
    drive.component_name = "something"
    code = Code(drive=drive, name="sample.tar.gz")

    with open("file.py", "w") as f:
        f.write('raise Exception("An error")')

    with tarfile.open("sample.tar.gz", "w:gz") as tar:
        tar.add("file.py")

    drive.put("sample.tar.gz")
    os.remove("file.py")
    os.remove("sample.tar.gz")

    python_script = TracerPythonScript("file.py", script_args=["--b=1"], raise_exception=False, code=code)
    run_work_isolated(python_script, params={"--a": "1"}, restart_count=0)
    assert "An error" in python_script.status.message

    with open("file.py", "w") as f:
        f.write("import sys\n")
        f.write("print(sys.argv)\n")

    with tarfile.open("sample.tar.gz", "w:gz") as tar:
        tar.add("file.py")

    _set_work_context()
    drive.put("sample.tar.gz")
    os.remove("file.py")
    os.remove("sample.tar.gz")

    with open("file.py", "w") as f:
        f.write('raise Exception("An error")')

    call_hash = python_script._calls[CacheCallsKeys.LATEST_CALL_HASH]
    python_script._calls[call_hash]["statuses"].pop(-1)
    python_script._calls[call_hash]["statuses"].pop(-1)

    run_work_isolated(python_script, params={"--a": "1"}, restart_count=1)
    assert python_script.has_succeeded
    assert python_script.script_args == ["--b=1", "--a=1"]
    os.remove("file.py")
    os.remove("sample.tar.gz")


def test_tracer_component_with_code_in_dir(tmp_path):
    """This test ensures the Tracer Component gets the latest code from the code object that is provided and arguments
    are cleaned."""

    drive = Drive("lit://code")
    drive.component_name = "something"
    code = Code(drive=drive, name="sample.tar.gz")

    with open("file.py", "w") as f:
        f.write('raise Exception("An error")')

    with tarfile.open("sample.tar.gz", "w:gz") as tar:
        tar.add("file.py")

    drive.put("sample.tar.gz")
    os.remove("file.py")
    os.remove("sample.tar.gz")

    python_script = TracerPythonScript("file.py", script_args=["--b=1"], raise_exception=False, code=code)
    run_work_isolated(python_script, params={"--a": "1"}, restart_count=0, code_dir=str(tmp_path))
    assert "An error" in python_script.status.message

    assert os.path.exists(os.path.join(str(tmp_path), "file.py"))
