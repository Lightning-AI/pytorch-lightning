from unittest import mock, TestCase

from lightning_app.utilities.log_helpers import _error_callback


class TestErrorCallback(TestCase):
    def test_known_error(self):
        websocket = mock.Mock()
        with self.assertLogs("lightning_app.utilities.log_helpers") as captured:
            _error_callback(websocket, ValueError())
        # check that there is only one log message
        self.assertEqual(len(captured.records), 1)
        # and it contains the error message expected
        self.assertIn("Error while reading logs (Malformed date format)", captured.records[0].getMessage())

    def test_unknown_error(self):
        websocket = mock.Mock()
        with self.assertLogs("lightning_app.utilities.log_helpers") as captured:
            _error_callback(websocket, IOError())
        # check that there is only one log message
        self.assertEqual(len(captured.records), 1)
        # and it contains the error message expected
        self.assertIn("Error while reading logs (Unknown)", captured.records[0].getMessage())
