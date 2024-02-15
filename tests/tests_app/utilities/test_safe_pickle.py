import subprocess
from pathlib import Path


def test_safe_pickle_app():
    test_dir = Path(__file__).parent / "testdata"
    proc = subprocess.Popen(
        ["lightning_app", "run", "app", "safe_pickle_app.py", "--open-ui", "false"],
        stdout=subprocess.PIPE,
        cwd=test_dir,
    )
    stdout, _ = proc.communicate()
    assert "Exiting the pickling app successfully" in stdout.decode("UTF-8")
