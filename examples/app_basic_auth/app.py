from typing import Optional

from fastapi import FastAPI, Header
from uvicorn import run

from lightning import LightningApp, LightningFlow, LightningWork


class Server(LightningWork):
    def run(self):
        app = FastAPI()

        @app.get("/")
        def read_root(x_lightning_user_id: Optional[str] = Header(None)):
            return {"Hello": x_lightning_user_id}

        run(app, host=self.host, port=self.port)


class Flow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.server = Server()

    def run(self):
        self.server.run()

    def configure_layout(self):
        return {"name": "Server with Auth", "content": self.server}


app = LightningApp(Flow())
