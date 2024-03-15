from unittest import TestCase, mock

from lightning.app.utilities.log_helpers import _error_callback


class TestErrorCallback(TestCase):
    def test_known_error(self):
        websocket = mock.Mock()
        with self.assertLogs("lightning.app.utilities.log_helpers") as captured:
            _error_callback(websocket, ValueError())
        # check that there is only one log message
        assert len(captured.records) == 1
        # and it contains the error message expected
        assert "Error while reading logs (Malformed date format)" in captured.records[0].getMessage()

    def test_unknown_error(self):
        websocket = mock.Mock()
        with self.assertLogs("lightning.app.utilities.log_helpers") as captured:
            _error_callback(websocket, OSError())
        # check that there is only one log message
        assert len(captured.records) == 1
        # and it contains the error message expected
        assert "Error while reading logs (Unknown)" in captured.records[0].getMessage()
