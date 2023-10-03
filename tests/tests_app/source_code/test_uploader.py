from unittest import mock
from unittest.mock import ANY, MagicMock

import pytest
from lightning.app.source_code import uploader

# keeping as global var so individual tests can access/modify it
response = {"response": MagicMock(headers={"ETag": "test-etag"})}


class MockedRequestSession(MagicMock):
    def put(self, url, data):
        assert url == "https://test-url"
        assert data == "test-data"
        return response["response"]

    def mount(self, prefix, adapter):
        assert prefix == "https://"
        assert adapter.max_retries.total == 10


@mock.patch("builtins.open", mock.mock_open(read_data="test-data"))
@mock.patch("lightning.app.source_code.uploader.requests.Session", MockedRequestSession)
def test_file_uploader():
    file_uploader = uploader.FileUploader(
        presigned_url="https://test-url", source_file="test.txt", total_size=100, name="test.txt"
    )
    file_uploader.progress = MagicMock()

    file_uploader.upload()

    file_uploader.progress.add_task.assert_called_once_with("upload", filename="test.txt", total=100)
    file_uploader.progress.start.assert_called_once()
    file_uploader.progress.update.assert_called_once_with(ANY, advance=9)


@mock.patch("builtins.open", mock.mock_open(read_data="test-data"))
@mock.patch("lightning.app.source_code.uploader.requests.Session", MockedRequestSession)
def test_file_uploader_failing_when_no_etag():
    response["response"] = MagicMock(headers={})
    presigned_url = "https://test-url"
    file_uploader = uploader.FileUploader(
        presigned_url=presigned_url, source_file="test.txt", total_size=100, name="test.txt"
    )
    file_uploader.progress = MagicMock()

    with pytest.raises(ValueError, match=f"Unexpected response from {presigned_url}, response"):
        file_uploader.upload()
