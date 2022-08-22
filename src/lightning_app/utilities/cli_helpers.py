import re
from typing import Dict, Optional

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

    metadata = {"tag": tag, "parameters": {}}

    if cls_path:
        metadata["cls_path"] = cls_path

    if cls_name:
        metadata["cls_name"] = cls_name

    if not parameters:
        return metadata

    metadata["parameters"].update({d["name"]: d["schema"]["type"] for d in parameters})
    return metadata


def _extract_command_from_openapi(openapi_resp: Dict) -> Dict[str, Dict[str, str]]:
    command_paths = [p for p in openapi_resp["paths"] if p.startswith("/command/")]
    return {p.replace("/command/", ""): _get_metadata_from_openapi(openapi_resp["paths"], p) for p in command_paths}


def _retrieve_application_url_and_available_commands(app_id_or_name_or_url: Optional[str]):
    """This function is used to retrieve the current url associated with an id."""

    if _is_url(app_id_or_name_or_url):
        url = app_id_or_name_or_url
        assert url
        resp = requests.get(url + "/openapi.json")
        if resp.status_code != 200:
            raise Exception(f"The server didn't process the request properly. Found {resp.json()}")
        return url, _extract_command_from_openapi(resp.json())

    # 2: If no identifier has been provided, evaluate the local application
    failed_locally = False

    if app_id_or_name_or_url is None:
        try:
            url = f"http://localhost:{APP_SERVER_PORT}"
            resp = requests.get(f"{url}/openapi.json")
            if resp.status_code != 200:
                raise Exception(f"The server didn't process the request properly. Found {resp.json()}")
            return url, _extract_command_from_openapi(resp.json())
        except requests.exceptions.ConnectionError:
            failed_locally = True

    # 3: If an identified was provided or the local evaluation has failed, evaluate the cloud.
    if app_id_or_name_or_url or failed_locally:
        client = LightningClient()
        project = _get_project(client)
        list_lightningapps = client.lightningapp_instance_service_list_lightningapp_instances(project.project_id)

        lightningapp_names = [lightningapp.name for lightningapp in list_lightningapps.lightningapps]

        if not app_id_or_name_or_url:
            raise Exception(f"Provide an application name, id or url with --app_id=X. Found {lightningapp_names}")

        for lightningapp in list_lightningapps.lightningapps:
            if lightningapp.id == app_id_or_name_or_url or lightningapp.name == app_id_or_name_or_url:
                if lightningapp.status.url == "":
                    raise Exception("The application is starting. Try in a few moments.")
                resp = requests.get(lightningapp.status.url + "/openapi.json")
                if resp.status_code != 200:
                    raise Exception(f"The server didn't process the request properly. Found {resp.json()}")
                return lightningapp.status.url, _extract_command_from_openapi(resp.json())
    return None, None
