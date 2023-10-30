import argparse
import os

import uvicorn
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Server Parser")
    parser.add_argument("--filepath", type=str, help="Where to find the `filepath`")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host`")
    parser.add_argument("--port", type=int, default="8888", help="Server port`")
    hparams = parser.parse_args()

    fastapi_service = FastAPI()

    if not os.path.exists(str(hparams.filepath)):
        content = ["The file wasn't transferred"]
    else:
        with open(hparams.filepath) as fo:
            content = fo.readlines()  # read the file received from SourceWork.

    @fastapi_service.get("/file")
    async def get_file_content(request: Request, response_class=HTMLResponse):
        lines = "\n".join(["<p>" + line + "</p>" for line in content])
        return HTMLResponse(f"<html><head></head><body><ul>{lines}</ul></body></html>")

    uvicorn.run(app=fastapi_service, host=hparams.host, port=hparams.port)
