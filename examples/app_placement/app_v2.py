from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run

from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork


class Work(LightningWork):
    def __init__(self, **kwargs):
        super().__init__(parallel=True, **kwargs)

    def run(self):
        fastapi_service = FastAPI()

        fastapi_service.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @fastapi_service.get(f"/{self.name}")
        def get_root_name():
            return {"Hello Word!"}

        @fastapi_service.get("/")
        def get_root():
            return {"Hello Word!"}

        run(fastapi_service, host=self.host, port=self.port)


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
        return [{"name": "w_" + str(i), "content": w} for i, w in enumerate(self.works())]


app = LightningApp(Flow(), debug=True)
