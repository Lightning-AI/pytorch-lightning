from json import JSONDecodeError, loads
from typing import Any

from click import ClickException, Context, Group
from lightning_cloud.openapi.rest import ApiException


class ApiExceptionHandler(Group):
    """Attempts to convert ApiExceptions to ClickExceptions.

    This process clarifies the error for the user by:
    1. Showing the error message from the lightning.ai servers,
       instead of showing the entire HTTP response
    2. Suppressing long tracebacks

    However, if the ApiException cannot be decoded, or is not
    a 4xx error, the original ApiException will be re-raised.
    """

    def invoke(self, ctx: Context) -> Any:
        try:
            return super().invoke(ctx)
        except ApiException as api:
            exception_messages = []
            if 400 <= api.status < 500:
                try:
                    body = loads(api.body)
                except JSONDecodeError:
                    raise api
                exception_messages.append(body["message"])
                exception_messages.extend(body["details"])
            else:
                raise api
            raise ClickException("\n".join(exception_messages))
