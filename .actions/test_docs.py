# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from unittest.mock import MagicMock

import pytest
import requests
from assistant import adjust_linked_external_docs


@pytest.fixture
def version_lookup(monkeypatch):
    lookup = MagicMock(return_value=["3.2.1"])
    monkeypatch.setattr("lightning_utilities.docs.formatting._load_pypi_versions", lookup)
    return lookup


@pytest.fixture
def page(tmp_path):
    path = tmp_path / "guide.rst"
    path.write_text("Guide <https://docs.example/stable/guide.html>\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("status", [404, 503])
def test_unavailable_version_keeps_stable(page, version_lookup, monkeypatch, caplog, status):
    response = MagicMock()
    response.__enter__.return_value = response
    response.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status}")
    monkeypatch.setattr(requests, "get", MagicMock(return_value=response))
    original = page.read_bytes()

    adjust_linked_external_docs(
        "https://docs.example/stable/",
        "https://docs.example/{example.__version__}/",
        str(page.parent),
        version_digits=3,
    )

    assert page.read_bytes() == original
    assert "keeping" in caplog.text.lower()


@pytest.mark.parametrize("error", [requests.Timeout, requests.ConnectionError])
def test_unreachable_version_keeps_stable(page, version_lookup, monkeypatch, error):
    monkeypatch.setattr(requests, "get", MagicMock(side_effect=error("Unavailable")))
    original = page.read_bytes()

    adjust_linked_external_docs(
        "https://docs.example/stable/",
        "https://docs.example/{example.__version__}/",
        str(page.parent),
        version_digits=3,
    )

    assert page.read_bytes() == original


def test_available_version_rewrites_links(page, version_lookup, monkeypatch):
    response = MagicMock()
    response.__enter__.return_value = response
    get = MagicMock(return_value=response)
    monkeypatch.setattr(requests, "get", get)

    adjust_linked_external_docs(
        "https://docs.example/stable/",
        "https://docs.example/{example.__version__}/",
        str(page.parent),
        version_digits=3,
    )

    assert page.read_text(encoding="utf-8") == "Guide <https://docs.example/3.2.1/guide.html>\n"
    get.assert_called_once_with("https://docs.example/3.2.1/", timeout=10, stream=True)
    version_lookup.assert_called_once_with("example")


def test_available_version_preserves_rewrite_options(page, version_lookup, monkeypatch):
    response = MagicMock()
    response.__enter__.return_value = response
    get = MagicMock(return_value=response)
    monkeypatch.setattr(requests, "get", get)
    other = page.with_suffix(".txt")
    other.write_bytes(page.read_bytes())

    adjust_linked_external_docs(
        "https://docs.example/stable/",
        "https://docs.example/{example.__version__}/",
        [str(page.parent)],
        file_extensions=(".txt",),
    )

    assert "https://docs.example/3.2/guide.html" in other.read_text(encoding="utf-8")
    assert "https://docs.example/stable/guide.html" in page.read_text(encoding="utf-8")
    get.assert_called_once_with("https://docs.example/3.2/", timeout=10, stream=True)
