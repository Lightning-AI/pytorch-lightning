import json
import logging
import os
import sys
import tempfile
from typing import List

import requests
import torch

from lightning_app.model2cloud.authentication import authenticate
from lightning_app.model2cloud.save import (
    _download_and_extract_data_to,
    _save_checkpoint_from_path,
    _save_meta_data,
    _save_model,
    _save_model_code,
    _save_model_weights,
    _save_requirements_file,
    _submit_data_to_url,
    _write_and_save_requirements,
    get_linked_output_dir,
)
from lightning_app.model2cloud.utils import (
    get_model_data,
    LIGHTNING_CLOUD_URL,
    LIGHTNING_STORAGE_FILE,
    split_name,
    stage,
)

if os.getenv("LIGHTNING_MODEL_STORE_TESTING", 0):
    from tests.constants import LIGHTNING_TEST_STORAGE_DIR as LIGHTNING_STORAGE_DIR
else:
    from lightning_app.model2cloud.utils import LIGHTNING_STORAGE_DIR

import lightning as L
import pytorch_lightning as PL

logging.basicConfig(level=logging.INFO)


def to_lightning_cloud(
    name: str,
    version: str = "latest",
    model=None,
    source_code_path: str = "",
    checkpoint_path: str = "",
    requirements_file_path: str = "",
    requirements: List[str] = [],
    weights_only: bool = False,
    api_key: str = "",
    project_id: str = "",
    progress_bar: bool = True,
    save_code: bool = True,
    *args,
    **kwargs,
):
    """
    Parameters
    ==========

    :param: name: (str)
        The model name. Model/Checkpoint will be uploaded with this unique name. Format: "model_name"
    :param: version: (str, default="latest")
        The version of the model to be uploaded. If not provided, default will be latest (not overridden).
    :param: model: (default=None)
        The model object (initialized). This is optional, but if `checkpoint_path` is not passed,
            it will raise an error. (Optional)
    :param: source_code_path: (str, default="")
        The path to the source code that needs to be uploaded along with the model.
        The path can point to a python file or a directory. Path pointing to a non-python file
            will raise an error. (Optional)
    :param: checkpoint_path (str, default="")
        The path to the checkpoint that needs to be uploaded. (Optional)
    :param: requirements_file_path (str, default="")
        The path to the requirements file, will always be uploaded with the name of `requirements.txt`.
    :param: requirements (List[str], default=[])
        List of requirements as strings, that will be written as `requirements.txt` and
            then uploaded. If both `requirements_file_path` and `requirements` are passed,
            a warning is raised as `requirements` is given the priority over `requirements_file_path`.
    :param: weights_only (bool, default=False)
        If set to `True`, it will only save model weights and nothing else. This raises
            an error if `weights_only` is `True` but no `model` is passed.
    :param: api_key (str, default="")
        API_KEY used for authentication. Fetch it after logging to https://lightning.ai
            (in the keys tab in the settings). If not passed, the API will attempt to
            either find the credentials in your system or opening the login prompt.
    :param: project_id (str, default="")
        Some users have multiple projects with unique `project_id`. They need to pass
            this in order to upload models to the cloud.
    :param: progress_bar (bool, default=True)
        A progress bar to show the uploading status. Disable this if not needed, by setting to `False`.
    :param: save_code (bool, default=True)
        By default, the API saves the code where the model is defined.
        Set it to `False` if saving code is not desired.

    Returns
    ========
    None
    """
    if model is None and checkpoint_path is None:
        raise ValueError(
            """"
            You either need to pass the model or the checkpoint path that you want to save. :)
            Any one of: `to_lightning_cloud("model_name", model=modelObj, ...)`
            or `to_lightning_cloud("model_name", checkpoint_path="your_checkpoint_path.ckpt", ...)`
            is required.
            """
        )

    if weights_only and not model:
        raise ValueError(
            "No model passed to `to_lightning_cloud(...), in order to save weights,"
            " you need to pass the model object."
        )

    version = version or "latest"
    _, model_name, _ = split_name(name, version=version, l_stage=stage.UPLOAD)
    username_from_api_key, api_key = authenticate(api_key)

    name = f"{username_from_api_key}/{model_name}:{version}"

    stored = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        if checkpoint_path:
            stored = _save_checkpoint_from_path(
                model_name,
                path=checkpoint_path,
                tmpdir=tmpdir,
                stored=stored,
            )

        if model:
            stored = _save_model_weights(
                model_name,
                model_state_dict=model.state_dict(),
                tmpdir=tmpdir,
                stored=stored,
                *args,
                **kwargs,
            )

            if not weights_only:
                stored = _save_model(
                    model_name,
                    model=model,
                    tmpdir=tmpdir,
                    stored=stored,
                    *args,
                    **kwargs,
                )

                if save_code:
                    stored = _save_model_code(
                        model_name,
                        model_cls=model.__class__,
                        source_code_path=source_code_path,
                        tmpdir=tmpdir,
                        stored=stored,
                    )

        if requirements and requirements_file_path:
            # TODO: Later on, figure out how to merge requirements from both args
            logging.warning(
                "You provided a requirements file (requirements_file_path=...)"
                " and requirements list (requirements=...). In case of any collisions,"
                " anything that comes from requirements=... will be given the priority."
            )

        if requirements:
            stored = _write_and_save_requirements(
                model_name,
                requirements=requirements,
                stored=stored,
                tmpdir=tmpdir,
            )
        elif requirements_file_path:
            stored = _save_requirements_file(
                model_name,
                requirements_file_path=requirements_file_path,
                stored=stored,
                tmpdir=tmpdir,
            )

        url = _save_meta_data(
            model_name,
            stored=stored,
            version=version,
            model=model,
            username=username_from_api_key,
            api_key=api_key,
            project_id=project_id,
        )

        _submit_data_to_url(url, tmpdir, progress_bar=progress_bar)
        msg = "Finished storing the following data items to the Lightning Cloud.\n"
        for key, val in stored.items():
            if key == "code":
                if val["type"] == "file":
                    msg += f"Stored code as a file with name: {val['path']}\n"
                else:
                    msg += f"Stored code as a folder with name: {val['path']}\n"
            else:
                msg += f"Stored {key} with name: {val}\n"

        msg += (
            f'\nJust do: download_from_lightning_cloud("{username_from_api_key}/{model_name}", '
            f'version="{version}") in order to download the model from the cloud to your local system.'
        )
        msg += (
            f'\nAnd: to_lightning_cloud("{username_from_api_key}/{model_name}", '
            f'version="{version}") in order to load the downloaded model.'
        )
        logging.info(msg)


def _load_model(stored, output_dir, *args, **kwargs):
    if "model" in stored:
        sys.path.insert(0, f"{output_dir}")
        model = torch.load(f"{output_dir}/{stored['model']}", *args, **kwargs)
        return model
    else:
        raise ValueError(
            "Couldn't find the model when uploaded to our storage. Please check"
            " with the model owner to confirm that the models exist in the storage."
        )


def _load_weights(model, stored, output_dir, *args, **kwargs):
    if "weights" in stored:
        model.load_state_dict(torch.load(f"{output_dir}/{stored['weights']}", *args, **kwargs))
        return model
    else:
        raise ValueError(
            "Weights were not found, please contact the model owner to verify if the"
            " weights were stored successfully..."
        )


def _load_checkpoint(model, stored, output_dir, *args, **kwargs):
    if "checkpoint" in stored:
        ckpt = f"{output_dir}/{stored['checkpoint']}"
    else:
        raise ValueError(
            "No checkpoint path was found, please contact the model owner to verify if the"
            " checkpoint was saved successfully."
        )

    ckpt = model.load_from_checkpoint(ckpt, *args, **kwargs)
    return ckpt


def download_from_lightning_cloud(
    name: str,
    version: str = "latest",
    output_dir: str = "",
    progress_bar: bool = True,
):
    """
    Parameters
    ==========
    :param: name (str):
        The unique name of the model to be downloaded. Format: `<username>/<model_name>`.
    :param: version: (str, default="latest")
        The version of the model to be uploaded. If not provided, default will be latest (not overridden).
    :param: output_dir (str, default=""):
        The target directory, where the model and other data will be stored. If not passed,
            the data will be stored in `$HOME/.lightning/lightning_model_store/<username>/<model_name>/<version>`.
            (`version` defaults to `latest`)

    Returns
    =======
    None
    """
    version = version or "latest"
    username, model_name, version = split_name(name, version=version, l_stage=stage.DOWNLOAD)

    linked_output_dir = ""
    if not output_dir:
        output_dir = LIGHTNING_STORAGE_DIR
        output_dir = os.path.join(output_dir, username, model_name, version)
        linked_output_dir = get_linked_output_dir(output_dir)
    else:
        output_dir = os.path.abspath(output_dir)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    response = requests.get(f"{LIGHTNING_CLOUD_URL}/v1/models?name={username}/{model_name}&version={version}")
    assert response.status_code == 200, (
        f"Unable to download the model with name {name} and version {version}."
        " Maybe reach out to the model owner or check the arguments again?"
    )

    download_url_response = json.loads(response.content)
    download_url = download_url_response["downloadUrl"]
    meta_data = download_url_response["metadata"]

    logging.info(f"Downloading the model data for {name} to {output_dir} folder.")
    _download_and_extract_data_to(output_dir, download_url, progress_bar)

    if linked_output_dir:
        logging.info(f"Linking the downloaded folder from {output_dir} to {linked_output_dir} folder.")
        if os.path.islink(linked_output_dir):
            os.unlink(linked_output_dir)
        if os.path.exists(linked_output_dir):
            if os.path.isdir(linked_output_dir):
                os.rmdir(linked_output_dir)

        os.symlink(output_dir, linked_output_dir)

    with open(LIGHTNING_STORAGE_FILE, "w+") as storage_file:
        storage = {
            username: {
                model_name: {
                    version: {
                        "output_dir": output_dir,
                        "linked_output_dir": str(linked_output_dir),
                        "metadata": meta_data,
                    },
                },
            },
        }
        json.dump(storage, storage_file)

    logging.info("Downloading done...")
    logging.info(
        f"The source code for your model has been written to {output_dir} folder, and"
        f" linked to {linked_output_dir} folder."
    )

    logging.info(
        "Please make sure to add imports to the necessary classes needed for instantiation of"
        " your model before calling `load_from_lightning_cloud`."
    )


def _validate_output_dir(dir: str):
    if not os.path.exists(dir):
        raise ValueError(
            "The output directory doesn't exist... did you forget to call download_from_lightning_cloud(...)?"
        )


def load_from_lightning_cloud(
    name: str,
    version: str = "latest",
    load_weights: bool = False,
    load_checkpoint: bool = False,
    model=None,
    *args,
    **kwargs,
):
    """
    Parameters
    ==========
    :param: name (str)
        Name of the model to load. Format: `<username>/<model_name>`
    :param: version: (str, default="latest")
        The version of the model to be uploaded. If not provided, default will be latest (not overridden).
    :param: load_weights (bool, default=False)
        Loads only weights if this is set to `True`. Needs `model` to be passed in order to load the weights.
    :param: load_checkpoint (bool, default=False)
        Loads checkpoint if this is set to `True`. Only a `LightningModule` model is supported for this feature.

    Returns
    =======
    None
    """
    if load_weights and load_checkpoint:
        raise ValueError(
            f"You passed load_weights={load_weights} and load_checkpoint={load_checkpoint},"
            " it's expected that only one of them are requested in a single call."
        )

    if os.path.exists(LIGHTNING_STORAGE_FILE):
        version = version or "latest"
        model_data = get_model_data(name, version)
        output_dir = model_data["output_dir"]
        linked_output_dir = model_data["linked_output_dir"]
        meta_data = model_data["metadata"]
        stored = {"code": {}}

        for key, val in meta_data.items():
            if key.startswith("stored_"):
                if key.startswith("stored_code_"):
                    stored["code"][key.split("_code_")[1]] = val
                else:
                    stored[key.split("_")[1]] = val

        _validate_output_dir(output_dir)
        if linked_output_dir:
            _validate_output_dir(linked_output_dir)

        if load_weights:
            # This first loads the model - and then the weights
            if not model:
                raise ValueError(
                    "Expected model=... to be passed for loading weights, please pass"
                    f" your model object to load_from_lightning_cloud({name}, {version}, model=ModelObj)"
                )
            return _load_weights(model, stored, linked_output_dir or output_dir, *args, **kwargs)
        elif load_checkpoint:
            if not model:
                raise ValueError(
                    "You need to pass the LightningModule object (model) to be able to"
                    f" load the checkpoint. `load_from_lightning_cloud({name}, {version},"
                    " load_checkpoint=True, model=...)`"
                )
            if not isinstance(model, (PL.LightningModule, L.LightningModule)):
                raise TypeError(
                    "For loading checkpoints, the model is required to be a LightningModule"
                    f" or a subclass of LightningModule, got type {type(model)}."
                )

            return _load_checkpoint(model, stored, linked_output_dir or output_dir, *args, **kwargs)
        else:
            return _load_model(stored, linked_output_dir or output_dir, *args, **kwargs)
    else:
        raise ValueError(
            f"Could not find the model (for {name}:{version}) in the local system."
            " Did you make sure to download the model using: `download_from_lightning_cloud(...)`"
            " before calling `load_from_lightning_cloud(...)`?"
        )
