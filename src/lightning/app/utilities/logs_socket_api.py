# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import urllib
from typing import Callable, Optional
from urllib.parse import urlparse

from lightning_cloud.openapi import ApiClient, AuthServiceApi, V1LoginRequest
from websocket import WebSocketApp

from lightning.app.core import constants
from lightning.app.utilities.login import Auth


# This class joins common things for reading logs,
# initialization and getting API token
class _LogsSocketAPI:
    def __init__(self, api_client: ApiClient):
        self.api_client = api_client
        self._auth = Auth()
        self._auth.authenticate()
        self._auth_service = AuthServiceApi(api_client)

    def _get_api_token(self) -> str:
        token_resp = self._auth_service.auth_service_login(
            body=V1LoginRequest(
                username=self._auth.username,
                api_key=self._auth.api_key,
            )
        )
        return token_resp.token


class _LightningLogsSocketAPI(_LogsSocketAPI):
    @staticmethod
    def _app_logs_socket_url(host: str, project_id: str, app_id: str, token: str, component: str) -> str:
        return (
            f"wss://{host}/v1/projects/{project_id}/appinstances/{app_id}/logs?"
            f"token={token}&component={component}&follow=true"
        )

    def create_lightning_logs_socket(
        self,
        project_id: str,
        app_id: str,
        component: str,
        on_message_callback: Callable[[WebSocketApp, str], None],
        on_error_callback: Optional[Callable[[Exception, str], None]] = None,
    ) -> WebSocketApp:
        """Creates and returns WebSocketApp to listen to lightning app logs.

            .. code-block:: python
                # Synchronous reading, run_forever() is blocking


                def print_log_msg(ws_app, msg):
                    print(msg)


                flow_logs_socket = client.create_lightning_logs_socket("project_id", "app_id", "flow", print_log_msg)
                flow_socket.run_forever()

            .. code-block:: python
                # Asynchronous reading (with Threads)


                def print_log_msg(ws_app, msg):
                    print(msg)


                flow_logs_socket = client.create_lightning_logs_socket("project_id", "app_id", "flow", print_log_msg)
                work_logs_socket = client.create_lightning_logs_socket("project_id", "app_id", "work_1", print_log_msg)

                flow_logs_thread = Thread(target=flow_logs_socket.run_forever)
                work_logs_thread = Thread(target=work_logs_socket.run_forever)

                flow_logs_thread.start()
                work_logs_thread.start()
                # .......

                flow_logs_socket.close()
                work_logs_thread.close()

        Arguments:
            project_id: Project ID.
            app_id: Application ID.
            component: Component name eg flow.
            on_message_callback: Callback object which is called when received data.
            on_error_callback: Callback object which is called when we get error.

        Returns:
            WebSocketApp of the wanted socket
        """
        _token = self._get_api_token()
        clean_ws_host = urlparse(self.api_client.configuration.host).netloc
        socket_url = self._app_logs_socket_url(
            host=clean_ws_host,
            project_id=project_id,
            app_id=app_id,
            token=_token,
            component=component,
        )

        return WebSocketApp(socket_url, on_message=on_message_callback, on_error=on_error_callback)


class _ClusterLogsSocketAPI(_LogsSocketAPI):
    @staticmethod
    def _cluster_logs_socket_url(
        host: str, cluster_id: str, start: float, end: Optional[float], limit: int, token: str
    ) -> str:
        params = {
            "start": start,
            "limit": limit,
            "token": token,
        }

        if end:
            params["end"] = end
        else:
            params["follow"] = "true"

        return f"wss://{host}/v1/core/clusters/{cluster_id}/logs?{urllib.parse.urlencode(params)}"

    def create_cluster_logs_socket(
        self,
        cluster_id: str,
        start: float,  # unix timestamp
        end: Optional[float],  # unix timestamp
        limit: int,
        on_message_callback: Callable[[WebSocketApp, str], None],
        on_error_callback: Optional[Callable[[Exception, str], None]] = None,
    ) -> WebSocketApp:
        """Creates and returns WebSocketApp to listen to cluster logs.

            .. code-block:: python
                # Synchronous reading, run_forever() is blocking


                def print_log_msg(ws_app, msg):
                    print(msg)


                logs_socket = client.create_cluster_logs_socket("cluster_id", 1661100000, 1661101000, print_log_msg)
                logs_socket.run_forever()

            .. code-block:: python
                # Asynchronous reading (with Threads)


                def print_log_msg(ws_app, msg):
                    print(msg)


                logs_socket = client.create_cluster_logs_socket("cluster_id", 1661100000, 1661101000, print_log_msg)

                logs_thread = Thread(target=cluster_logs_socket.run_forever)

                logs_thread.start()
                # .......


                logs_socket.close()

        Arguments:
            cluster_id: Project ID.
            start: Starting timestamp to query cluster logs from.
            end: Ending timestamp to query cluster logs to.
            limit: A maximal number of log lines to get.
            on_message_callback: Callback object which is called when received data.
            on_error_callback: Callback object which is called when we get error.

        Returns:
            WebSocketApp of the wanted socket
        """
        _token = self._get_api_token()
        clean_ws_host = urlparse(self.api_client.configuration.host).netloc
        socket_url = self._cluster_logs_socket_url(
            host=clean_ws_host,
            cluster_id=cluster_id,
            token=_token,
            start=start,
            limit=limit,
            end=end,
        )

        if constants.DEBUG:
            print(f"Connecting to websocket {socket_url}")

        return WebSocketApp(socket_url, on_message=on_message_callback, on_error=on_error_callback)
