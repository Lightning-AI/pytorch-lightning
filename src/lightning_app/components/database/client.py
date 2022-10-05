from typing import Any, Dict, List, Optional, Type, TypeVar

import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from lightning_app.components.database.utilities import GeneralModel

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
    def __init__(self, db_url: str, model: Optional[T] = None):
        self.db_url = db_url
        self.model = model
        self._session = None

    def select_all(self, model: Optional[Type[T]] = None) -> List[T]:
        cls = model if model else self.model
        assert cls
        resp = self.session.post(self.db_url + "/select_all/", data=GeneralModel.from_cls(cls).json())
        assert resp.status_code == 200
        return [cls(**data) for data in resp.json()]

    def insert(self, model: T):
        resp = self.session.post(
            self.db_url + "/insert/",
            data=GeneralModel.from_obj(model).json(),
        )
        assert resp.status_code == 200

    def update(self, model: T) -> None:
        resp = self.session.post(
            self.db_url + "/update/",
            data=GeneralModel.from_obj(model).json(),
        )
        assert resp.status_code == 200

    def delete(self, model: T) -> None:
        resp = self.session.post(
            self.db_url + "/delete/",
            data=GeneralModel.from_obj(model).json(),
        )
        assert resp.status_code == 200

    def delete_database(self) -> None:
        resp = self.session.post(self.db_url + "/delete_database/")
        assert resp.status_code == 200

    def reset_database(self) -> None:
        resp = self.session.post(self.db_url + "/reset_database/")
        assert resp.status_code == 200

    @property
    def session(self):
        if self._session is None:
            self._session = _configure_session()
        return self._session

    def to_dict(self) -> Dict[str, Any]:
        return {"db_url": self.db_url, "model": self.model.__name__ if self.model else None}
