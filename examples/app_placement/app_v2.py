from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run

from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork


def healthz():
    """Health check endpoint used in the cloud FastAPI servers to check the status periodically."""
    return {"status": "ok"}


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
        # trailing / is required for urljoin to properly join the path. In case of
        # multiple trailing /, urljoin removes them
        fastapi_service.get("/healthz", status_code=200)(healthz)

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
        return [{"name": w.name, "content": w} for w in self.works()]


app = LightningApp(Flow(), debug=True)
