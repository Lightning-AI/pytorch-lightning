from tests_pytorch.helpers.runif import RunIf


@RunIf(standalone=True)
def test_standalone_command():
    assert hex(17) == "0x11"
