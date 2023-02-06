import os
import sys

import pytest

from lightning.app.cli.commands.cd import _CD_FILE, cd
from lightning.app.cli.commands.pwd import pwd


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows yet")
def test_cd():
    """This test validates cd behaves as expected."""
    assert "/" == cd("/")
    assert "/" == pwd()
    assert "/hero" == cd("hero")
    assert "/hero" == pwd()
    assert f"/hero{os.sep}something_else" == cd("something_else")
    assert f"/hero{os.sep}something_else" == pwd()
    assert f"/hero{os.sep}something_else{os.sep}hello{os.sep}a" == cd("hello/a")
    assert f"/hero{os.sep}something_else{os.sep}hello{os.sep}a" == pwd()
    assert f"/hero{os.sep}something_else" == cd(f"..{os.sep}..")
    assert f"/hero{os.sep}something_else" == pwd()
    assert "/hero" == cd("..")
    assert "/hero" == pwd()
    assert "/" == cd("/")
    assert "/" == pwd()
    assert "/a" == cd("../a")
    assert "/a" == pwd()
    assert f"/a{os.sep}thomas{os.sep}hello" == cd(f"thomas{os.sep}hello")
    assert f"/a{os.sep}thomas{os.sep}hello" == pwd()
    assert f"/thomas{os.sep}hello" == cd(f"/thomas{os.sep}hello")
    assert f"/thomas{os.sep}hello" == pwd()
    assert "/" == cd("/")
    assert "/name with spaces" == cd("name with spaces")
    assert "/name with spaces/name with spaces 2" == cd("name with spaces 2")

    os.remove(_CD_FILE)
