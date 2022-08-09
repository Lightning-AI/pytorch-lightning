import os

import pytest
from tests_app import _PROJECT_ROOT

from lightning_app.components.python import PopenPythonScript, TracerPythonScript
from lightning_app.testing.helpers import RunIf
from lightning_app.testing.testing import run_work_isolated

COMPONENTS_SCRIPTS_FOLDER = str(os.path.join(_PROJECT_ROOT, "tests/tests_app/components/python/scripts/"))


def test_non_existing_python_script():
    match = "tests/components/python/scripts/0.py"
    with pytest.raises(FileNotFoundError, match=match):
        python_script = PopenPythonScript(match)
        run_work_isolated(python_script)
        assert not python_script.has_started

    with pytest.raises(FileNotFoundError, match=match):
        python_script = TracerPythonScript(match)
        run_work_isolated(python_script)
        assert not python_script.has_started


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


@RunIf(skip_windows=True)
def test_popen_python_script_failure():
    python_script = PopenPythonScript(
        COMPONENTS_SCRIPTS_FOLDER + "c.py",
        env={"VARIABLE": "1"},
        raise_exception=False,
    )
    run_work_isolated(python_script)
    assert python_script.has_failed
    assert python_script.status.message == "1"


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
