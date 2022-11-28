import subprocess
from pathlib import Path


def test_safe_pickle_app():
    test_dir = Path(__file__).parent / "testdata"
    proc = subprocess.Popen(
        ["lightning", "run", "app", "safe_pickle_app.py", "--open-ui", "false"], stdout=subprocess.PIPE, cwd=test_dir
    )
    assert "Exiting the pickling app successfully!!" in proc.stdout.read().decode("UTF-8")
