import os
import subprocess

import pytest

import lightning_app as la
from lightning_app.cli import cmd_init, cmd_react_ui_init
from lightning_app.testing.helpers import RunIf


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") is None, reason="not running in GH actions.")
@pytest.mark.skip(reason="need to figure out how to mock not having npm")
def test_missing_npm():
    with pytest.raises(SystemExit, match="This machine is missing 'npm'"):
        cmd_react_ui_init._check_react_prerequisites()


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") is None, reason="not running in GH actions.")
@pytest.mark.skip(reason="need to figure out how to mock not having node")
def test_missing_nodejs():
    with pytest.raises(SystemExit, match="This machine is missing 'node'"):
        cmd_react_ui_init._check_react_prerequisites()


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") is None, reason="not running in GH actions")
@pytest.mark.skip(reason="need to figure out how to mock not having yarn")
def test_missing_yarn():
    with pytest.raises(SystemExit, match="This machine is missing 'yarn'"):
        cmd_react_ui_init._check_react_prerequisites()


@RunIf(skip_windows=True)
def test_copy_and_setup_react_ui(tmpdir):
    dest_dir = os.path.join(tmpdir, "react-ui")
    subprocess.Popen(["python", "-m", "lightning", "init", "react-ui", "--dest_dir", dest_dir]).wait()

    # make sure package is minimal
    files = sorted(f for f in os.listdir(dest_dir) if f != "__pycache__")
    assert len(files) == 3, "should only be 3 objects: readme.md, example_app.py and ui dir"

    # make sure index.html has the vite app placeholder
    index_content = open(dest_dir + "/ui/dist/index.html").read()
    assert "<title>Vite App</title>" in index_content

    # read the compiled js file
    js_file = [x for x in os.listdir(os.path.join(dest_dir, "ui", "dist", "assets")) if ".js" in x]
    js_file = os.path.join(dest_dir, f"ui/dist/assets/{js_file[0]}")
    index_content = open(js_file).read()

    # if this is in the compiled file, the compilation worked and the app will work
    assert "Total number of prints in your terminal:" in index_content, "react app was not compiled properly"
    assert "LightningState.subscribe" in index_content, "react app was not compiled properly"


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") is None, reason="not running in GH actions")
def test_correct_num_react_template_files():
    template_dir = os.path.join(la.__path__[0], "cli/react-ui-template")
    files = cmd_init._ls_recursively(template_dir)
    assert len(files) == 15, "react-ui template files must be minimal... do not add nice to haves"
