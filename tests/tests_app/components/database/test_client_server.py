import os
from pathlib import Path
from typing import Optional

import pytest

from lightning import LightningApp
from lightning_app import LightningFlow
from lightning_app.components.database import Database
from lightning_app.components.database.utilities import GeneralModel
from lightning_app.runners import MultiProcessRuntime
from lightning_app.utilities.imports import _is_sqlmodel_available

if _is_sqlmodel_available():
    from sqlmodel import Field, SQLModel

    class TestConfig(SQLModel, table=True):
        __table_args__ = {"extend_existing": True}

        id: Optional[int] = Field(default=None, primary_key=True)
        name: str


@pytest.mark.parametrize("mode", ["flow", "work"])
@pytest.mark.skipif(not _is_sqlmodel_available(), reason="sqlmodel is required for this test.")
def test_client_server(mode):

    database_path = Path("database.db").resolve()
    if database_path.exists():
        os.remove(database_path)

    general = GeneralModel.from_obj(TestConfig(name="name"))
    assert general.cls_name == "TestConfig"
    assert general.cls_module == "test_client_server"
    assert general.data == '{"id": null, "name": "name"}'

    class Flow(LightningFlow):
        def __init__(self):
            super().__init__()
            self.db = Database(models=[TestConfig], mode=mode)
            self.tracker = None

        def run(self):
            self.db.run()

            if not self.db.alive():
                return

            if self.tracker is None:
                self.db.insert(TestConfig(name="name"))
                elem: TestConfig = self.db.select_all(TestConfig)[0]
                assert elem.name == "name"
                self.tracker = "update"

            elif self.tracker == "update":
                elem: TestConfig = self.db.select_all(TestConfig)[0]
                elem.name = "new_name"
                self.db.update(elem)

                elem: TestConfig = self.db.select_all(TestConfig)[0]
                assert elem.name == "new_name"
                self.tracker = "delete"

            elif self.tracker == "delete":
                elem: TestConfig = self.db.select_all(TestConfig)[0]
                elem: TestConfig = self.db.delete(elem)

                assert not self.db.select_all(TestConfig)
                self.db.insert(TestConfig(name="name"))

                assert self.db.select_all(TestConfig)

                self.db.reset_database()
                assert len(self.db.select_all(TestConfig)) == 0

                self.db.delete_database()
                self._exit()

    app = LightningApp(Flow())
    MultiProcessRuntime(app, start_server=False).dispatch()

    assert not database_path.exists()
