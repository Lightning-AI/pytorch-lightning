import os
from functools import partial
from typing import List, Optional, Type, Union

from fastapi import FastAPI
from uvicorn import run

from lightning import BuildConfig, LightningFlow, LightningWork
from lightning_app.components.database.client import _DatabaseClientFlow, _DatabaseClientWork
from lightning_app.components.database.utilities import (
    create_database,
    delete_database,
    general_delete,
    general_insert,
    general_select_all,
    general_update,
    reset_database,
)
from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.imports import _is_sqlmodel_available

if _is_sqlmodel_available():
    from sqlmodel import SQLModel


logger = Logger(__name__)
engine = None


class DatabaseType:
    FLOW = "flow"
    WORK = "work"


class Database(LightningFlow):
    def __init__(
        self,
        models: Union[Type["SQLModel"], List[Type["SQLModel"]]],
        db_filename: str = "database.db",
        debug: bool = False,
        mode: str = DatabaseType.WORK,
    ):
        """The Database Component enables to interact with an SQLite database to store some structured information
        about your application.

        The provided models are SQLModel tables

        Arguments:
            models: A SQLModel or a list of SQLModels table to be added to the database.
            db_filename: The name of the SQLite database.
            debug: Whether to run the database in debug mode.
            mode: Whether the database should be running within the flow or dedicated work.

        Example:

            from sqlmodel import SQLModel, Field
            from lightning import LightningFlow, LightningApp
            from lightning_app.components.database import Database

            class CounterModel(SQLModel, table=True):
                __table_args__ = {"extend_existing": True}

                id: int = Field(default=None, primary_key=True)
                count: int


            class Flow(LightningFlow):

                def __init__(self):
                    super().__init__()
                    self.db = Database(models=[CounterModel])
                    self.counter = 0

                def run(self):
                    self.db.run()

                    if not self.db.alive():
                        return

                    if self.counter == 0:
                        self.db.reset_database()

                    rows = self.db.select_all(CounterModel)
                    print(f"{self.counter}: {rows}")
                    if not rows:
                        self.db.insert(CounterModel(count=0))
                    else:
                        row : CounterModel= rows[0]
                        row.count += 1
                        self.db.update(row)

                    self.counter += 1

            app = LightningApp(Flow())
        """
        super().__init__()
        self.mode = mode
        self.db_filename = db_filename
        self._models = models if isinstance(models, list) else [models]
        if self.mode == DatabaseType.WORK:
            self.database_server_work = _DatabaseServerWork(self._models, db_filename=db_filename, debug=debug)
        else:
            create_database(db_filename, self._models, debug)
        self._client = None

    def run(self):
        if self.mode == DatabaseType.WORK:
            self.database_server_work.run()

    def alive(self) -> bool:
        if self.mode == DatabaseType.WORK:
            return self.database_server_work.alive()
        else:
            return True

    def insert(self, model: "SQLModel"):
        return self.client.insert(model)

    def update(self, model: "SQLModel"):
        return self.client.update(model)

    def select_all(self, model: Type["SQLModel"]):
        return self.client.select_all(model)

    def delete(self, model: "SQLModel"):
        return self.client.delete(model)

    @property
    def client(self):
        if not self.alive():
            return

        if not self._client:
            if self.mode == DatabaseType.WORK:
                self._client = _DatabaseClientWork(db_url=self.database_server_work.db_url)
            else:
                self._client = _DatabaseClientFlow(self.db_filename)
        return self._client

    def delete_database(self):
        if self.client:
            if self.mode == DatabaseType.WORK:
                self.client._delete_database()
            else:
                delete_database(self.db_filename, self._models)
        else:
            raise Exception("The database isn't ready yet.")

    def reset_database(self):
        if self.client:
            if self.mode == DatabaseType.WORK:
                self.client._reset_database()
            else:
                reset_database(self.db_filename, self._models)
        else:
            raise Exception("The database isn't ready yet.")


class _DatabaseServerWork(LightningWork):
    def __init__(
        self,
        models: Union[Type["SQLModel"], List[Type["SQLModel"]]],
        db_filename: str = "database.db",
        debug: bool = False,
    ):
        super().__init__(parallel=True, cloud_build_config=BuildConfig(["sqlmodel"]))
        self.db_filename = db_filename
        self.debug = debug
        self._models = models if isinstance(models, list) else [models]
        self._client = None

    def run(self):
        app = FastAPI()

        create_database(self.db_filename, self._models, self.debug)
        app.get("/general/")(general_select_all)
        app.post("/general/")(general_insert)
        app.put("/general/")(general_update)
        app.delete("/general/")(general_delete)
        app.post("/delete_database/")(partial(delete_database, self.db_filename, self._models))
        app.post("/reset_database/")(partial(reset_database, self.db_filename, self._models))

        run(app, host=self.host, port=self.port, log_level="error")

    def alive(self):
        """Hack: Returns whether the server is alive."""
        return self.db_url != ""

    @property
    def db_url(self) -> Optional[str]:
        use_localhost = "LIGHTNING_APP_STATE_URL" not in os.environ
        if use_localhost:
            return self.url
        if self.internal_ip != "":
            return f"http://{self.internal_ip}:{self.port}"
        return self.internal_ip
