import os

from lightning.app.cli.commands.cd import _CD_FILE, cd
from lightning.app.cli.commands.pwd import pwd


def test_cd():
    """This test validates cd behaves as expected."""
    assert "/" == cd("/")
    assert "/" == pwd()
    assert "/hero" == cd("hero")
    assert "/hero" == pwd()
    assert "/hero/something_else" == cd("something_else")
    assert "/hero/something_else" == pwd()
    assert "/hero/something_else/hello/a" == cd("hello/a")
    assert "/hero/something_else/hello/a" == pwd()
    assert "/hero/something_else" == cd("../..")
    assert "/hero/something_else" == pwd()
    assert "/hero" == cd("..")
    assert "/hero" == pwd()
    assert "/" == cd("/")
    assert "/" == pwd()
    assert "/a" == cd("../a")
    assert "/a" == pwd()
    assert "/a/thomas/hello" == cd("thomas/hello")
    assert "/a/thomas/hello" == pwd()
    assert "/thomas/hello" == cd("/thomas/hello")
    assert "/thomas/hello" == pwd()

    os.remove(_CD_FILE)
