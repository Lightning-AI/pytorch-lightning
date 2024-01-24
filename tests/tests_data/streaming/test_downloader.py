import os
from unittest.mock import MagicMock

from lightning.data.streaming.downloader import S3Downloader, subprocess


def test_s3_downloader_fast(tmpdir, monkeypatch):
    monkeypatch.setattr(os, "system", MagicMock(return_value=0))
    popen_mock = MagicMock()
    monkeypatch.setattr(subprocess, "Popen", MagicMock(return_value=popen_mock))
    downloader = S3Downloader(tmpdir, tmpdir, [])
    downloader.download_file("s3://random_bucket/a.txt", os.path.join(tmpdir, "a.txt"))
    popen_mock.wait.assert_called()
