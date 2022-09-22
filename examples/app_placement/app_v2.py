from fastapi import FastAPI
from uvicorn import run

from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork


class Work(LightningWork):
    def __init__(self, **kwargs):
        super().__init__(parallel=True, **kwargs)

    def run(self):
        fastapi_app = FastAPI()

        @fastapi_app.get("/")
        def get_root():
            return {"Hello Word!"}

        run(fastapi_app, host=self.host, port=self.port)


class Flow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.cloud_compute = CloudCompute(name="cpu-small")
        self.work_a = Work()
        self.work_b = Work()
        self.work_c = Work(cloud_compute=self.cloud_compute)

    def run(self):
        for work in self.works():
            work.run()

        if all(w.has_succeeded for w in self.works()):
            self._exit("Application End !")

    def configure_layout(self):
        return [{"name": w.name, "content": w} for w in self.works()]


app = LightningApp(Flow(), debug=True)
