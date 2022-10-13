import asyncio
import os
import sys
from typing import List, Optional, Type, Union

import uvicorn
from fastapi import FastAPI
from uvicorn import run

from lightning import BuildConfig, LightningWork
from lightning_app.components.database.utilities import create_database, Delete, Insert, SelectAll, Update
from lightning_app.storage import Drive
from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.imports import _is_sqlmodel_available

if _is_sqlmodel_available():
    from sqlmodel import SQLModel


logger = Logger(__name__)
engine = None


# Required to avoid Uvicorn Server overriding Lightning App signal handlers.
# Discussions: https://github.com/encode/uvicorn/discussions/1708
class DatabaseUvicornServer(uvicorn.Server):

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
            mode: Whether the database should be running within the flow or dedicated work.
            token: Token used to protect the database access. Ensure you don't expose it through the App State.

        Example:

            from sqlmodel import SQLModel, Field
            from lightning import LightningFlow, LightningApp
            from lightning_app.components.database import Database, DatabaseClient

            class CounterModel(SQLModel, table=True):
                __table_args__ = {"extend_existing": True}

                id: int = Field(default=None, primary_key=True)
                count: int


            class Flow(LightningFlow):

                def __init__(self):
                    super().__init__()
                    self.db = Database(models=[CounterModel])
                    self._client = None
                    self.counter = 0

                def run(self):
                    self.db.run()

                    if not self.db.alive():
                        return

                    if self.counter == 0:
                        self._client = DatabaseClient(model=CounterModel, db_url=self.db.url)
                        self._client.reset_database()

                    assert self._client

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
                        self._client.delete_database()
                        self._exit()

                    self.counter += 1

            app = LightningApp(Flow())
        """
        super().__init__(parallel=True, cloud_build_config=BuildConfig(["sqlmodel"]))
        self.db_filename = db_filename
        self.debug = debug
        self._models = models if isinstance(models, list) else [models]
        self.drive = Drive("lit://database")

    def run(self, token: Optional[str] = None) -> None:
        """
        Arguments:
            token: Token used to protect the database access. Ensure you don't expose it through the App State.
        """
        if self.drive.list():
            self.drive.get(self.db_filename)
            print("Retrieved the database from Drive.")

        app = FastAPI()

        create_database(self.db_filename, self._models, self.debug)
        models = {m.__name__: m for m in self._models}
        app.post("/select_all/")(SelectAll(models, token))
        app.post("/insert/")(Insert(models, token))
        app.post("/update/")(Update(models, token))
        app.post("/delete/")(Delete(models, token))

        sys.modules["uvicorn.main"].Server = DatabaseUvicornServer

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
