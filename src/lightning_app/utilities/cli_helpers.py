import json
import os
import re
import sys
from typing import Dict, Optional

import arrow
import click
import requests

from lightning_app.core.constants import APP_SERVER_PORT
from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.network import LightningClient


def _format_input_env_variables(env_list: tuple) -> Dict[str, str]:
    """
    Args:
        env_list:
           List of str for the env variables, e.g. ['foo=bar', 'bla=bloz']

    Returns:
        Dict of the env variables with the following format
            key: env variable name
            value: env variable value
    """

    env_vars_dict = {}
    for env_str in env_list:
        var_parts = env_str.split("=")
        if len(var_parts) != 2 or not var_parts[0]:
            raise Exception(
                f"Invalid format of environment variable {env_str}, "
                f"please ensure that the variable is in the format e.g. foo=bar."
            )
        var_name, value = var_parts

        if var_name in env_vars_dict:
            raise Exception(f"Environment variable '{var_name}' is duplicated. Please only include it once.")

        if not re.match(r"[0-9a-zA-Z_]+", var_name):
            raise ValueError(
                f"Environment variable '{var_name}' is not a valid name. It is only allowed to contain digits 0-9, "
                f"letters A-Z, a-z and _ (underscore)."
            )

        env_vars_dict[var_name] = value
    return env_vars_dict


def _is_url(id: Optional[str]) -> bool:
    if isinstance(id, str) and (id.startswith("https://") or id.startswith("http://")):
        return True
    return False


def _get_metadata_from_openapi(paths: Dict, path: str):
    parameters = paths[path]["post"].get("parameters", {})
    tag = paths[path]["post"].get("tags", [None])[0]
    cls_path = paths[path]["post"].get("cls_path", None)
    cls_name = paths[path]["post"].get("cls_name", None)
    description = paths[path]["post"].get("description", None)
    requirements = paths[path]["post"].get("requirements", None)

    metadata = {"tag": tag, "parameters": {}}

    if cls_path:
        metadata["cls_path"] = cls_path

    if cls_name:
        metadata["cls_name"] = cls_name

    if description:
        metadata["description"] = description

    if description:
        metadata["requirements"] = requirements

    if not parameters:
        return metadata

    metadata["parameters"].update({d["name"]: d["schema"]["type"] for d in parameters})
    return metadata


def _extract_command_from_openapi(openapi_resp: Dict) -> Dict[str, Dict[str, str]]:
    command_paths = [p for p in openapi_resp["paths"] if p.startswith("/command/")]
    return {p.replace("/command/", ""): _get_metadata_from_openapi(openapi_resp["paths"], p) for p in command_paths}


class _LightningAppOpenAPIRetriever:
    def __init__(
        self,
        app_id_or_name_or_url: Optional[str],
        use_cache: bool = False,
    ):
        """This class encapsulates the logic to collect the openapi.json file from the app to use the CLI Commands.

        Arguments:
            app_id_or_name_or_url: An identified for the app.
            use_cache: Whether to load the openapi spec from the cache.
        """
        self.app_id_or_name_or_url = app_id_or_name_or_url
        self.url = None
        self.openapi = None
        self.api_commands = None
        self.app_id = None
        home = os.path.expanduser("~")
        if use_cache:
            cache_openapi = os.path.join(home, ".lightning", "lightning_connection", "commands", "openapi.json")
            if os.path.exists(cache_openapi):
                with open(cache_openapi) as f:
                    self.openapi = json.load(f)
                self.api_commands = _extract_command_from_openapi(self.openapi)

        if not self.api_commands:
            self._collect_open_api_json()
            if self.openapi:
                self.api_commands = _extract_command_from_openapi(self.openapi)

    def is_alive(self) -> bool:
        """Returns whether the Lightning App Rest API is available."""
        if self.url is None:
            self._maybe_find_url()
        if self.url is None:
            return False
        resp = requests.get(self.url)
        return resp.status_code == 200

    def _maybe_find_url(self):
        """Tries to resolve the app url from the provided `app_id_or_name_or_url`."""
        if _is_url(self.app_id_or_name_or_url):
            self.url = self.app_id_or_name_or_url
            assert self.url
            return

        if self.app_id_or_name_or_url is None:
            url = f"http://localhost:{APP_SERVER_PORT}"
            resp = requests.get(f"{self.url}/openapi.json")
            if resp.status_code == 200:
                self.url = url
                return

        app = self._maybe_find_matching_cloud_app()
        if app:
            self.url = app.status.url

    def _maybe_find_matching_cloud_app(self):
        """Tries to resolve the app url from the provided `app_id_or_name_or_url`."""
        client = LightningClient()
        project = _get_project(client)
        list_apps = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project.project_id)

        app_names = [lightningapp.name for lightningapp in list_apps.lightningapps]

        if not self.app_id_or_name_or_url:
            print(f"ERROR: Provide an application name, id or url with --app_id=X. Found {app_names}")
            sys.exit(0)

        for app in list_apps.lightningapps:
            if app.id == self.app_id_or_name_or_url or app.name == self.app_id_or_name_or_url:
                if app.status.url == "":
                    print("The application is starting. Try in a few moments.")
                    sys.exit(0)
                return app

    def _collect_open_api_json(self):
        """This function is used to retrieve the current url associated with an id."""

        if _is_url(self.app_id_or_name_or_url):
            self.url = self.app_id_or_name_or_url
            assert self.url
            resp = requests.get(self.url + "/openapi.json")
            if resp.status_code != 200:
                print(f"ERROR: The server didn't process the request properly. Found {resp.json()}")
                sys.exit(0)
            self.openapi = resp.json()
            return

        # 2: If no identifier has been provided, evaluate the local application
        if self.app_id_or_name_or_url is None:
            try:
                self.url = f"http://localhost:{APP_SERVER_PORT}"
                resp = requests.get(f"{self.url}/openapi.json")
                if resp.status_code != 200:
                    raise Exception(f"The server didn't process the request properly. Found {resp.json()}")
                self.openapi = resp.json()
            except requests.exceptions.ConnectionError:
                pass

        # 3: If an identified was provided or the local evaluation has failed, evaluate the cloud.
        else:
            app = self._maybe_find_matching_cloud_app()
            if app:
                if app.status.url == "":
                    raise Exception("The application is starting. Try in a few moments.")
                resp = requests.get(app.status.url + "/openapi.json")
                if resp.status_code != 200:
                    raise Exception(
                        "The server didn't process the request properly. " "Try once your application is ready."
                    )
                self.url = app.status.url
                self.openapi = resp.json()
                self.app_id = app.id


def _arrow_time_callback(
    _ctx: "click.core.Context", _param: "click.core.Option", value: str, arw_now=arrow.utcnow()
) -> arrow.Arrow:
    try:
        return arw_now.dehumanize(value)
    except ValueError:
        try:
            return arrow.get(value)
        except (ValueError, TypeError):
            raise click.ClickException(f"cannot parse time {value}")
