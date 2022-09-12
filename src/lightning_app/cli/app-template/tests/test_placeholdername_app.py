r"""
To test a lightning app:
1. Use LightningTestApp which is a subclass of LightningApp.
2. Subclass run_once in LightningTestApp.
3. in run_once, come up with a way to verify the behavior you wanted.

run_once runs your app through one cycle of the event loop and then terminates
"""
import io
import os
from contextlib import redirect_stdout

from lightning.app.testing.testing import application_testing, LightningTestApp


class LightningAppTestInt(LightningTestApp):
    def run_once(self) -> bool:
        f = io.StringIO()
        with redirect_stdout(f):
            super().run_once()
        out = f.getvalue()
        assert out == "hello from component A\nhello from component B\n"
        return True


def test_templatename_app():
    start_dir = os.getcwd()
    os.chdir("..")

    cwd = os.getcwd()
    cwd = os.path.join(cwd, "placeholdername/app.py")
    command_line = [
        cwd,
        "--blocking",
        "False",
        "--open-ui",
        "False",
    ]
    result = application_testing(LightningAppTestInt, command_line)
    assert result.exit_code == 0

    # reset dir
    os.chdir(start_dir)
