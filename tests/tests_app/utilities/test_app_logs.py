from datetime import datetime
from time import sleep
from unittest.mock import MagicMock

from lightning.app.utilities.app_logs import _LogEvent


def test_log_event():
    event_1 = _LogEvent("", datetime.now(), MagicMock(), MagicMock())
    sleep(0.1)
    event_2 = _LogEvent("", datetime.now(), MagicMock(), MagicMock())
    assert event_1 < event_2
    assert event_1 <= event_2
