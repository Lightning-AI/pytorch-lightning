import multiprocessing as mp
from argparse import ArgumentParser
from typing import Optional
from urllib.parse import urljoin

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from lightning_app.frontend.frontend import Frontend
from lightning_app.utilities.log import get_frontend_logfile
from lightning_app.utilities.network import find_free_network_port


class StaticWebFrontend(Frontend):
    """A frontend that serves static files from a directory using FastAPI.

    Return this in your `LightningFlow.configure_layout()` method if you wish to serve a HTML page.

    Arguments:
        serve_dir: A local directory to serve files from. This directory should at least contain a file `index.html`.

    Example:

        In your LightningFlow, override the method `configure_layout`:

        .. code-block:: python

            def configure_layout(self):
                return StaticWebFrontend("path/to/folder/to/serve")
    """

    def __init__(self, serve_dir: str) -> None:
        super().__init__()
        self.serve_dir = serve_dir
        self._process: Optional[mp.Process] = None

    def start_server(self, host: str, port: int) -> None:
        log_file = str(get_frontend_logfile())
        self._process = mp.Process(
            target=start_server,
            kwargs=dict(
                host=host,
                port=port,
                serve_dir=self.serve_dir,
                path=f"/{self.flow.name}",
                log_file=log_file,
            ),
        )
        self._process.start()

    def stop_server(self) -> None:
        if self._process is None:
            raise RuntimeError("Server is not running. Call `StaticWebFrontend.start_server()` first.")
        self._process.kill()


def healthz():
    """Health check endpoint used in the cloud FastAPI servers to check the status periodically."""
    return {"status": "ok"}


def start_server(serve_dir: str, host: str = "localhost", port: int = -1, path: str = "/", log_file: str = "") -> None:
    if port == -1:
        port = find_free_network_port()
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
    fastapi_service.get(urljoin(f"{path}/", "healthz"), status_code=200)(healthz)
    fastapi_service.mount(path, StaticFiles(directory=serve_dir, html=True), name="static")

    log_config = _get_log_config(log_file) if log_file else uvicorn.config.LOGGING_CONFIG

    uvicorn.run(app=fastapi_service, host=host, port=port, log_config=log_config)


def _get_log_config(log_file: str) -> dict:
    """Returns a logger configuration in the format expected by uvicorn that sends all logs to the given logfile."""
    # Modified from the default config found in uvicorn.config.LOGGING_CONFIG
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(message)s",
                "use_colors": False,
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.FileHandler",
                "filename": log_file,
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
        },
    }


if __name__ == "__main__":  # pragma: no-cover
    parser = ArgumentParser()
    parser.add_argument("serve_dir", type=str)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=-1)
    args = parser.parse_args()
    start_server(serve_dir=args.serve_dir, host=args.host, port=args.port)
