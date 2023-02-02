import os

from lightning.app.cli.commands.cd import _CD_FILE, cd


def test_cd():
    """This test validates cd behaves as expected."""
    assert "/" == cd("/")
    assert "/hero" == cd("hero")
    assert "/hero/something_else" == cd("something_else")
    assert "/hero/something_else/hello/a" == cd("hello/a")
    assert "/hero/something_else" == cd("../..")
    assert "/hero" == cd("..")
    assert "/" == cd("/")
    assert "/a" == cd("../a")
    assert "/a/thomas/hello" == cd("thomas/hello")
    assert "/thomas/hello" == cd("/thomas/hello")

    os.remove(_CD_FILE)
