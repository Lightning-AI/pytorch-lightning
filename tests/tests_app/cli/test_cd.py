import os
import sys
from unittest import mock

import pytest
from lightning.app.cli.commands import cd
from lightning.app.cli.commands.pwd import pwd


@mock.patch("lightning.app.cli.commands.cd.ls", mock.MagicMock())
@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows yet")
def test_cd(monkeypatch):
    """This test validates cd behaves as expected."""
    ls = mock.MagicMock()
    monkeypatch.setattr(cd, "ls", ls)

    assert cd.cd("/") == "/"
    assert pwd() == "/"
    ls.ls.return_value = ["hero"]
    assert cd.cd("hero") == "/hero"
    assert pwd() == "/hero"
    ls.ls.return_value = ["something_else"]
    assert f"/hero{os.sep}something_else" == cd.cd("something_else")
    assert f"/hero{os.sep}something_else" == pwd()
    ls.ls.return_value = ["hello"]
    assert f"/hero{os.sep}something_else{os.sep}hello{os.sep}a" == cd.cd("hello/a")
    assert f"/hero{os.sep}something_else{os.sep}hello{os.sep}a" == pwd()
    assert f"/hero{os.sep}something_else" == cd.cd(f"..{os.sep}..")
    ls.ls.return_value = ["something_else"]
    assert f"/hero{os.sep}something_else" == pwd()
    assert cd.cd("..") == "/hero"
    assert pwd() == "/hero"
    assert cd.cd("/") == "/"
    assert pwd() == "/"
    ls.ls.return_value = ["a"]
    assert cd.cd("../a") == "/a"
    assert pwd() == "/a"
    ls.ls.return_value = ["thomas"]
    assert f"/a{os.sep}thomas{os.sep}hello" == cd.cd(f"thomas{os.sep}hello")
    assert f"/a{os.sep}thomas{os.sep}hello" == pwd()
    ls.ls.return_value = ["thomas"]
    assert f"/thomas{os.sep}hello" == cd.cd(f"/thomas{os.sep}hello")
    assert f"/thomas{os.sep}hello" == pwd()
    assert cd.cd("/") == "/"
    ls.ls.return_value = ["name with spaces"]
    assert cd.cd("name with spaces") == "/name with spaces"
    ls.ls.return_value = ["name with spaces 2"]
    assert cd.cd("name with spaces 2") == "/name with spaces/name with spaces 2"

    os.remove(cd._CD_FILE)

    mock_exit = mock.MagicMock()
    monkeypatch.setattr(cd, "_error_and_exit", mock_exit)
    assert cd.cd("/") == "/"
    ls.ls.return_value = ["project_a"]
    cd.cd("project_b")
    assert mock_exit._mock_call_args.args[0] == "no such file or directory: /project_b"
