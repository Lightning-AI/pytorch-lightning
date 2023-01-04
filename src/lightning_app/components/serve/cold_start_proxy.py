import asyncio
from typing import Any

import aiohttp
from fastapi import HTTPException
from pydantic import BaseModel

from diffusion_with_autoscaler.datatypes import Image, Text

proxy_url = "https://ulhcn-01gd3c9epmk5xj2y9a9jrrvgt8.litng-ai-03.litng.ai/api/predict"


class ColdStartProxy:
    """ColdStartProxy allows users to configure the load balancer to use a proxy service while the work is cold
    starting. This is useful with services that gets realtime requests but startup time for workers is high.

    If the request body is same and the method is POST for the proxy service,
    then the default implementation of `handle_request` can be used. In that case
    initialize the proxy with the proxy url. Otherwise, the user can override the `handle_request`

    Args:
        proxy_url (str): The url of the proxy service
    """

    def __init__(self, proxy_url):
        self.proxy_url = proxy_url
        self.proxy_timeout = 50
        # checking `asyncio.iscoroutinefunction` instead of `inspect.iscoroutinefunction`
        # because AsyncMock in the tests requres the former to pass
        if not asyncio.iscoroutinefunction(self.handle_request):
            raise TypeError("handle_request must be an `async` function")

    async def handle_request(self, request: BaseModel) -> Any:
        """This method is called when the request is received while the work is cold starting. The default
        implementation of this method is to forward the request body to the proxy service with POST method but the
        user can override this method to handle the request in any way.

        Args:
            request (BaseModel): The request body, a pydantic model that is being
            forwarded by load balancer which is a FastAPI service
        """
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "accept": "application/json",
                    "Content-Type": "application/json",
                }
                async with session.post(
                        self.proxy_url,
                        json=request.dict(),
                        timeout=self.proxy_timeout,
                        headers=headers,
                ) as response:
                    return await response.json()
        except Exception as ex:
            # TODO - test this and make sure if cold start proxy is not up,
            #  we are returning a useful message to the user
            raise HTTPException(status_code=500, detail=f"Error in proxy: {ex}")


class CustomColdStartProxy(ColdStartProxy):
    async def handle_request(self, request: Text) -> Any:
        async with aiohttp.ClientSession() as session:
            headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
            }
            async with session.post(
                    self.proxy_url,
                    json={"prompt": request.text},
                    timeout=self.proxy_timeout,
                    headers=headers,
            ) as response:
                resp = await response.json()
                return Image(image=resp["image"][22:])
