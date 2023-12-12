from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork
from uvicorn import run


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

        @fastapi_service.get("/")
        def get_root():
            return {"Hello Word!"}

        run(fastapi_service, host=self.host, port=self.port)


class Flow(LightningFlow):
    def __init__(self):
        super().__init__()
        # In the Cloud: All the works defined without passing explicitly a CloudCompute object
        # are running on the default machine.
        # This would apply to `work_a`, `work_b` and the dynamically created `work_d`.

        self.work_a = Work()
        self.work_b = Work()

        self.work_c = Work(cloud_compute=CloudCompute(name="cpu-small"))

    def run(self):
        if not hasattr(self, "work_d"):
            self.work_d = Work()

        for work in self.works():
            work.run()

    def configure_layout(self):
        return [{"name": w.name, "content": w} for i, w in enumerate(self.works())]


app = LightningApp(Flow(), log_level="debug")
