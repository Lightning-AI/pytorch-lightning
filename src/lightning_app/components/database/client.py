from typing import Optional, Type

import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from lightning_app.components.database.utilities import (
    general_delete,
    general_insert,
    general_select_all,
    general_update,
    GeneralModel,
)
from lightning_app.utilities.imports import _is_sqlmodel_available

if _is_sqlmodel_available():
    from sqlmodel import SQLModel

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


class _DatabaseClientWork:
    def __init__(self, db_url: str, model: Optional[Type["SQLModel"]] = None):
        self.db_url = db_url
        self.model = model
        self.session = _configure_session()

    def select_all(self, model: Optional[Type["SQLModel"]] = None):
        cls = model if model else self.model
        assert cls
        resp = self.session.get(self.db_url + "/general/", data=GeneralModel.from_cls(cls).json())
        assert resp.status_code == 200
        return [cls(**data) for data in resp.json()]

    def insert(self, model: "SQLModel"):
        resp = self.session.post(
            self.db_url + "/general/",
            data=GeneralModel.from_obj(model).json(),
        )
        assert resp.status_code == 200

    def update(self, model: "SQLModel"):
        resp = self.session.put(
            self.db_url + "/general/",
            data=GeneralModel.from_obj(model).json(),
        )
        assert resp.status_code == 200

    def delete(self, model: "SQLModel"):
        resp = self.session.delete(
            self.db_url + "/general/",
            data=GeneralModel.from_obj(model).json(),
        )
        assert resp.status_code == 200

    def _delete_database(self):
        resp = self.session.post(self.db_url + "/delete_database/")
        assert resp.status_code == 200

    def _reset_database(self):
        resp = self.session.post(self.db_url + "/reset_database/")
        assert resp.status_code == 200


class _DatabaseClientFlow:
    def __init__(self, db_filename: str, model: Optional[Type["SQLModel"]] = None):
        self.model = model

    def select_all(self, model: Optional[Type["SQLModel"]] = None):
        cls = model if model else self.model
        return general_select_all(model=GeneralModel.from_cls(cls))

    def insert(self, model: "SQLModel"):
        return general_insert(GeneralModel.from_obj(model))

    def update(self, model: "SQLModel"):
        return general_update(GeneralModel.from_obj(model))

    def delete(self, model: "SQLModel"):
        return general_delete(GeneralModel.from_obj(model))
