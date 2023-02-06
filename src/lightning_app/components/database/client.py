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

from typing import Any, Dict, List, Optional, Type, TypeVar

import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from lightning_app.components.database.utilities import _GeneralModel

_CONNECTION_RETRY_TOTAL = 5
_CONNECTION_RETRY_BACKOFF_FACTOR = 1


def _configure_session() -> Session:
    """Configures the session for GET and POST requests.

    It enables a generous retrial strategy that waits for the application server to connect.
    """
    retry_strategy = Retry(
        # wait time between retries increases exponentially according to: backoff_factor * (2 ** (retry - 1))
        total=_CONNECTION_RETRY_TOTAL,
        backoff_factor=_CONNECTION_RETRY_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)
    return http


T = TypeVar("T")


class DatabaseClient:
    def __init__(self, db_url: str, token: Optional[str] = None, model: Optional[T] = None) -> None:
        self.db_url = db_url
        self.model = model
        self.token = token or ""
        self._session = None

    def select_all(self, model: Optional[Type[T]] = None) -> List[T]:
        cls = model if model else self.model
        resp = self.session.post(
            self.db_url + "/select_all/", data=_GeneralModel.from_cls(cls, token=self.token).json()
        )
        assert resp.status_code == 200
        return [cls(**data) for data in resp.json()]

    def insert(self, model: T) -> None:
        resp = self.session.post(
            self.db_url + "/insert/",
            data=_GeneralModel.from_obj(model, token=self.token).json(),
        )
        assert resp.status_code == 200

    def update(self, model: T) -> None:
        resp = self.session.post(
            self.db_url + "/update/",
            data=_GeneralModel.from_obj(model, token=self.token).json(),
        )
        assert resp.status_code == 200

    def delete(self, model: T) -> None:
        resp = self.session.post(
            self.db_url + "/delete/",
            data=_GeneralModel.from_obj(model, token=self.token).json(),
        )
        assert resp.status_code == 200

    @property
    def session(self):
        if self._session is None:
            self._session = _configure_session()
        return self._session

    def to_dict(self) -> Dict[str, Any]:
        return {"db_url": self.db_url, "model": self.model.__name__ if self.model else None}
