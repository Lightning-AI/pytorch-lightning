from sqlmodel import Field, SQLModel

from lightning import LightningApp, LightningFlow
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
