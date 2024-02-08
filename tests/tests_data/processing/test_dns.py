from lightning.data.processing.dns import optimize_dns_context


def test_optimize_dns_context(monkeypatch):

    with optimize_dns_context(True):
        pass
