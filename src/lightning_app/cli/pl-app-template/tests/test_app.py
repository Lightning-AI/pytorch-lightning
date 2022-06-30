import pytest


@pytest.mark.skip
def test_is_running_in_cloud(monkeypatch):
    from app import Main

    monkeypatch.setenv("LIGHTNING_CLOUD_APP_ID", "anything")
    app = Main()
    assert app.running_in_cloud

    monkeypatch.delenv("LIGHTNING_CLOUD_APP_ID", raising=False)
    app = Main()
    assert not app.running_in_cloud
