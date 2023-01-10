import asyncio
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

from lightning_app.utilities.imports import _is_aiohttp_available, requires

if _is_aiohttp_available():
    import aiohttp
    import aiohttp.client_exceptions


class ColdStartProxy:
    """ColdStartProxy allows users to configure the load balancer to use a proxy service while the work is cold
    starting. This is useful with services that gets realtime requests but startup time for workers is high.

    If the request body is same and the method is POST for the proxy service,
    then the default implementation of `handle_request` can be used. In that case
    initialize the proxy with the proxy url. Otherwise, the user can override the `handle_request`

    Args:
        proxy_url (str): The url of the proxy service
    """

    @requires(["aiohttp"])
    def __init__(self, proxy_url: str):
        self.proxy_url = proxy_url
        self.proxy_timeout = 50
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
            raise HTTPException(status_code=500, detail=f"Error in proxy: {ex}")
