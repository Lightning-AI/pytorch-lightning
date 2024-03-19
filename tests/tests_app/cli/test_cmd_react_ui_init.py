import os

import lightning.app as la
import pytest
from lightning.app.cli import cmd_init, cmd_react_ui_init
from lightning.app.testing.helpers import _RunIf


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") is None, reason="not running in GH actions.")
@pytest.mark.xfail(strict=False, reason="need to figure out how to mock not having npm")
def test_missing_npm():
    with pytest.raises(SystemExit, match="This machine is missing 'npm'"):
        cmd_react_ui_init._check_react_prerequisites()


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") is None, reason="not running in GH actions.")
@pytest.mark.xfail(strict=False, reason="need to figure out how to mock not having node")
def test_missing_nodejs():
    with pytest.raises(SystemExit, match="This machine is missing 'node'"):
        cmd_react_ui_init._check_react_prerequisites()


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") is None, reason="not running in GH actions")
@pytest.mark.xfail(strict=False, reason="need to figure out how to mock not having yarn")
def test_missing_yarn():
    with pytest.raises(SystemExit, match="This machine is missing 'yarn'"):
        cmd_react_ui_init._check_react_prerequisites()


@_RunIf(skip_windows=True)
def test_copy_and_setup_react_ui(tmpdir):
    dest_dir = os.path.join(tmpdir, "react-ui")
    os.system(f"lightning_app init react-ui --dest_dir={dest_dir}")

    # make sure package is minimal
    files = sorted(f for f in os.listdir(dest_dir) if f != "__pycache__")
    assert len(files) == 3, "should only be 3 objects: readme.md, example_app.py and ui dir"

    # make sure index.html has the vite app placeholder
    with open(dest_dir + "/ui/dist/index.html") as fo:
        index_content = fo.read()
    assert "<title>Vite App</title>" in index_content

    # read the compiled js file
    js_file = [x for x in os.listdir(os.path.join(dest_dir, "ui", "dist", "assets")) if ".js" in x]
    js_file = os.path.join(dest_dir, f"ui/dist/assets/{js_file[0]}")
    with open(js_file) as fo:
        index_content = fo.read()

    # if this is in the compiled file, the compilation worked and the app will work
    assert "Total number of prints in your terminal:" in index_content, "react app was not compiled properly"
    assert "LightningState.subscribe" in index_content, "react app was not compiled properly"


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") is None, reason="not running in GH actions")
def test_correct_num_react_template_files():
    template_dir = os.path.join(la.__path__[0], "cli/react-ui-template")
    files = cmd_init._ls_recursively(template_dir)
    # TODO: remove lock file!!!
    assert len(files) == 16, "react-ui template files must be minimal... do not add nice to haves"
