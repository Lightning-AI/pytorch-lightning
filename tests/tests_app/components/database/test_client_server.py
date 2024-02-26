import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from time import sleep
from typing import List, Optional
from uuid import uuid4

import pytest
from lightning.app import LightningApp, LightningFlow, LightningWork
from lightning.app.components.database import Database, DatabaseClient
from lightning.app.components.database.utilities import _GeneralModel, _pydantic_column_type
from lightning.app.runners import MultiProcessRuntime
from lightning.app.utilities.imports import _is_sqlmodel_available

if _is_sqlmodel_available():
    from sqlalchemy import Column
    from sqlmodel import Field, SQLModel

    class Secret(SQLModel):
        name: str
        value: str

    class TestConfig(SQLModel, table=True):
        __table_args__ = {"extend_existing": True}

        id: Optional[int] = Field(default=None, primary_key=True)
        name: str
        secrets: List[Secret] = Field(..., sa_column=Column(_pydantic_column_type(List[Secret])))


class Work(LightningWork):
    def __init__(self):
        super().__init__(parallel=True)
        self.done = False

    def run(self, client: DatabaseClient):
        rows = client.select_all()
        while len(rows) == 0:
            print(rows)
            sleep(0.1)
            rows = client.select_all()
        self.done = True


@pytest.mark.skipif(not _is_sqlmodel_available(), reason="sqlmodel is required for this test.")
def test_client_server():
    database_path = Path("database.db").resolve()
    if database_path.exists():
        os.remove(database_path)

    secrets = [Secret(name="example", value="secret")]

    general = _GeneralModel.from_obj(TestConfig(name="name", secrets=secrets), token="a")
    assert general.cls_name == "TestConfig"
    assert general.data == '{"id": null, "name": "name", "secrets": [{"name": "example", "value": "secret"}]}'

    class Flow(LightningFlow):
        def __init__(self):
            super().__init__()
            self._token = str(uuid4())
            self.db = Database(models=[TestConfig])
            self._client = None
            self.tracker = None
            self.work = Work()

        def run(self):
            self.db.run(token=self._token)

            if not self.db.alive():
                return

            if not self._client:
                self._client = DatabaseClient(model=TestConfig, db_url=self.db.url, token=self._token)

            assert self._client

            self.work.run(self._client)

            if self.tracker is None:
                self._client.insert(TestConfig(name="name", secrets=secrets))
                elem = self._client.select_all(TestConfig)[0]
                assert elem.name == "name"
                self.tracker = "update"
                assert isinstance(elem.secrets[0], Secret)
                assert elem.secrets[0].name == "example"
                assert elem.secrets[0].value == "secret"

            elif self.tracker == "update":
                elem = self._client.select_all(TestConfig)[0]
                elem.name = "new_name"
                self._client.update(elem)

                elem = self._client.select_all(TestConfig)[0]
                assert elem.name == "new_name"
                self.tracker = "delete"

            elif self.tracker == "delete" and self.work.done:
                self.work.stop()

                elem = self._client.select_all(TestConfig)[0]
                elem = self._client.delete(elem)

                assert not self._client.select_all(TestConfig)
                self._client.insert(TestConfig(name="name", secrets=secrets))

                assert self._client.select_all(TestConfig)
                self.stop()

    app = LightningApp(Flow())
    MultiProcessRuntime(app, start_server=False).dispatch()

    database_path = Path("database.db").resolve()
    if database_path.exists():
        os.remove(database_path)


@pytest.mark.skipif(sys.platform == "win32", reason="currently not supported for windows.")
@pytest.mark.skipif(not _is_sqlmodel_available(), reason="sqlmodel is required for this test.")
def test_work_database_restart():
    id = str(uuid4()).split("-")[0]

    class Flow(LightningFlow):
        def __init__(self, db_root=".", restart=False):
            super().__init__()
            self._db_filename = os.path.join(db_root, id)
            self.db = Database(db_filename=self._db_filename, models=[TestConfig])
            self._client = None
            self.restart = restart

        def run(self):
            self.db.run()

            if not self.db.alive():
                return
            if not self._client:
                self._client = DatabaseClient(self.db.db_url, None, model=TestConfig)

            if not self.restart:
                self._client.insert(TestConfig(name="echo", secrets=[Secret(name="example", value="secret")]))
                self.stop()
            else:
                assert os.path.exists(self._db_filename)
                assert len(self._client.select_all()) == 1
                self.stop()

    with tempfile.TemporaryDirectory() as tmpdir:
        app = LightningApp(Flow(db_root=tmpdir))
        MultiProcessRuntime(app).dispatch()

        # Note: Waiting for SIGTERM signal to be handled
        sleep(2)

        app = LightningApp(Flow(db_root=tmpdir, restart=True))
        MultiProcessRuntime(app).dispatch()

        # Note: Waiting for SIGTERM signal to be handled
        sleep(2)


@pytest.mark.skipif(sys.platform == "win32", reason="currently not supported for windows.")
@pytest.mark.skipif(not _is_sqlmodel_available(), reason="sqlmodel is required for this test.")
def test_work_database_periodic_store():
    id = str(uuid4()).split("-")[0]

    class Flow(LightningFlow):
        def __init__(self, db_root="."):
            super().__init__()
            self._db_filename = os.path.join(db_root, id)
            self.db = Database(db_filename=self._db_filename, models=[TestConfig], store_interval=1)
            self._client = None
            self._start_time = None
            self.counter = 0

        def run(self):
            self.counter += 1

            self.db.run()

            if not self.db.alive():
                return

            if not self._client:
                self._client = DatabaseClient(self.db.db_url, None, model=TestConfig)

            if self._start_time is None:
                self._client.insert(TestConfig(name="echo", secrets=[Secret(name="example", value="secret")]))
                self._start_time = time.time()

            elif (time.time() - self._start_time) > 2:
                assert os.path.exists(self._db_filename)
                assert len(self._client.select_all()) == 1
                self.stop()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            app = LightningApp(Flow(tmpdir))
            MultiProcessRuntime(app).dispatch()
    except Exception:
        print(traceback.print_exc())
