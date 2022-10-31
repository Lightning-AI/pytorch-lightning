import asyncio
import os
import sys
from typing import List, Optional, Type, Union

import uvicorn
from fastapi import FastAPI
from uvicorn import run

from lightning_app.components.database.utilities import _create_database, _Delete, _Insert, _SelectAll, _Update
from lightning_app.core.work import LightningWork
from lightning_app.storage import Drive
from lightning_app.utilities.imports import _is_sqlmodel_available
from lightning_app.utilities.packaging.build_config import BuildConfig

if _is_sqlmodel_available():
    from sqlmodel import SQLModel


# Required to avoid Uvicorn Server overriding Lightning App signal handlers.
# Discussions: https://github.com/encode/uvicorn/discussions/1708
class _DatabaseUvicornServer(uvicorn.Server):

    has_started_queue = None

    def run(self, sockets=None):
        self.config.setup_event_loop()
        loop = asyncio.get_event_loop()
        asyncio.ensure_future(self.serve(sockets=sockets))
        loop.run_forever()

    def install_signal_handlers(self):
        """Ignore Uvicorn Signal Handlers."""


class Database(LightningWork):
    def __init__(
        self,
        models: Union[Type["SQLModel"], List[Type["SQLModel"]]],
        db_filename: str = "database.db",
        debug: bool = False,
    ) -> None:
        """The Database Component enables to interact with an SQLite database to store some structured information
        about your application.

        The provided models are SQLModel tables

        Arguments:
            models: A SQLModel or a list of SQLModels table to be added to the database.
            db_filename: The name of the SQLite database.
            debug: Whether to run the database in debug mode.

        Example::

            from typing import List
            from sqlmodel import SQLModel, Field
            from uuid import uuid4

            from lightning_app import LightningFlow, LightningApp
            from lightning_app.components.database import Database, DatabaseClient

            class CounterModel(SQLModel, table=True):
                __table_args__ = {"extend_existing": True}

                id: int = Field(default=None, primary_key=True)
                count: int


            class Flow(LightningFlow):

                def __init__(self):
                    super().__init__()
                    self._private_token = uuid4().hex
                    self.db = Database(models=[CounterModel])
                    self._client = None
                    self.counter = 0

                def run(self):
                    self.db.run(token=self._private_token)

                    if not self.db.alive():
                        return

                    if self.counter == 0:
                        self._client = DatabaseClient(
                            model=CounterModel,
                            db_url=self.db.url,
                            token=self._private_token,
                        )

                    rows = self._client.select_all()

                    print(f"{self.counter}: {rows}")

                    if not rows:
                        self._client.insert(CounterModel(count=0))
                    else:
                        row: CounterModel = rows[0]
                        row.count += 1
                        self._client.update(row)

                    if self.counter >= 100:
                        row: CounterModel = rows[0]
                        self._client.delete(row)
                        self._exit()

                    self.counter += 1

            app = LightningApp(Flow())

        If you want to use nested SQLModels, we provide a utility to do so as follows:

        Example::

            from typing import List
            from sqlmodel import SQLModel, Field
            from sqlalchemy import Column

            from lightning_app.components.database.utilities import pydantic_column_type

            class KeyValuePair(SQLModel):
                name: str
                value: str

            class CounterModel(SQLModel, table=True):
                __table_args__ = {"extend_existing": True}

                name: int = Field(default=None, primary_key=True)

                # RIGHT THERE ! You need to use Field and Column with the `pydantic_column_type` utility.
                kv: List[KeyValuePair] = Field(..., sa_column=Column(pydantic_column_type(List[KeyValuePair])))
        """
        super().__init__(parallel=True, cloud_build_config=BuildConfig(["sqlmodel"]))
        self.db_filename = db_filename
        self.debug = debug
        self._models = models if isinstance(models, list) else [models]
        self.drive = None

    def run(self, token: Optional[str] = None) -> None:
        """
        Arguments:
            token: Token used to protect the database access. Ensure you don't expose it through the App State.
        """
        self.drive = Drive("lit://database")
        if self.drive.list(component_name=self.name):
            self.drive.get(self.db_filename)
            print("Retrieved the database from Drive.")

        app = FastAPI()

        _create_database(self.db_filename, self._models, self.debug)
        models = {m.__name__: m for m in self._models}
        app.post("/select_all/")(_SelectAll(models, token))
        app.post("/insert/")(_Insert(models, token))
        app.post("/update/")(_Update(models, token))
        app.post("/delete/")(_Delete(models, token))

        sys.modules["uvicorn.main"].Server = _DatabaseUvicornServer

        run(app, host=self.host, port=self.port, log_level="error")

    def alive(self) -> bool:
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

    def on_exit(self):
        self.drive.put(self.db_filename)
        print("Stored the database to the Drive.")
