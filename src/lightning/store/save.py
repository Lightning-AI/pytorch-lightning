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

import inspect
import json
import logging
import os
import shutil
import tarfile
from pathlib import Path, PurePath

import requests
import torch
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

from lightning.app.core.constants import LIGHTNING_MODELS_PUBLIC_REGISTRY

logging.basicConfig(level=logging.INFO)

_LIGHTNING_DIR = os.path.join(Path.home(), ".lightning")
__STORAGE_FILE_NAME = ".model_storage"
_LIGHTNING_STORAGE_FILE = os.path.join(_LIGHTNING_DIR, __STORAGE_FILE_NAME)
__STORAGE_DIR_NAME = "model_store"
_LIGHTNING_STORAGE_DIR = os.path.join(_LIGHTNING_DIR, __STORAGE_DIR_NAME)


def _check_id(id: str) -> str:
    if id[-1] != "/":
        id += "/"
    if id.count("/") != 2:
        raise ValueError("The format for the ID should be: <username>/<model_name or any id>")
    return id


def _save_checkpoint(name, checkpoint, tmpdir, stored: dict) -> dict:
    checkpoint_file_path = f"{tmpdir}/checkpoint.ckpt"
    torch.save(checkpoint, checkpoint_file_path)
    stored["checkpoint"] = "checkpoint.ckpt"
    return stored


def _save_checkpoint_from_path(name, path, tmpdir, stored: dict) -> dict:
    checkpoint_file_path = path
    shutil.copy(checkpoint_file_path, f"{tmpdir}/checkpoint.ckpt")
    stored["checkpoint"] = "checkpoint.ckpt"
    return stored


def _save_model_weights(name, model_state_dict, tmpdir, stored, *args, **kwargs) -> dict:
    # For now we assume that it's always going to be public
    weights_file_path = f"{tmpdir}/weights.pt"
    torch.save(model_state_dict, weights_file_path, *args, **kwargs)
    stored["weights"] = "weights.pt"
    return stored


def _save_model(name, model, tmpdir, stored, *args, **kwargs) -> dict:
    # For now we assume that it's always going to be public
    model_file_path = f"{tmpdir}/model"
    torch.save(model, model_file_path, *args, **kwargs)
    stored["model"] = "model"
    return stored


def _save_model_code(name, model_cls, source_code_path, tmpdir, stored) -> dict:
    if source_code_path:
        source_code_path = os.path.abspath(source_code_path)
        if not os.path.exists(source_code_path):
            raise FileExistsError(f"Given path {source_code_path} does not exist.")

        # Copy contents to tmpdir folder
        if os.path.isdir(source_code_path):
            logging.warning(
                f"NOTE: Folder: {source_code_path} is being uploaded to the cloud so that the user "
                " can make the necessary imports after downloading your model."
            )
            dir_name = os.path.basename(source_code_path)
            shutil.copytree(source_code_path, f"{tmpdir}/{dir_name}/")
            stored["code"] = {"type": "folder", "path": f"{dir_name}"}
        else:
            if os.path.splitext(source_code_path)[-1] != ".py":
                raise FileExistsError(
                    "Expected a Python file or a directory, to be uploaded for model definition,"
                    f" but found {source_code_path}. If your file is not a Python file, and you still"
                    " want to save it, please consider saving it in a folder and passing the folder"
                    " path instead."
                )

            logging.warning(
                f"NOTE: File: {source_code_path} is being uploaded to the cloud so that the"
                " user can make the necessary imports after downloading your model."
            )

            file_name = os.path.basename(source_code_path)
            shutil.copy(source_code_path, f"{tmpdir}/{file_name}")
            stored["code"] = {"type": "file", "path": f"{file_name}"}
    else:
        # TODO: Raise a warning if the file has any statements/expressions outside of
        # __name__ == "__main__" in the script
        # As those will be executed on import
        model_class_path = inspect.getsourcefile(model_cls)
        if model_class_path:
            if os.path.splitext(model_class_path)[-1] != ".py":
                raise FileExistsError(
                    f"The model definition was found in a non-python file ({model_class_path}),"
                    " which is not currently supported (for safety reasons). If your file is not a"
                    " Python file, and you still want to save it, please consider saving it in a"
                    " folder and passing the folder path instead."
                )

            file_name = os.path.basename(model_class_path)
            logging.warning(
                f"NOTE: File: {model_class_path} is being uploaded to the cloud so that the"
                " user can make the necessary imports after downloading your model. The file"
                f" will be saved as {file_name} on download."
            )
            shutil.copyfile(model_class_path, f"{tmpdir}/{file_name}")
            stored["code"] = {"type": "file", "path": file_name}
    return stored


def _write_and_save_requirements(name, requirements, stored, tmpdir):
    if not isinstance(requirements, list) and isinstance(requirements, str):
        requirements = [requirements]

    requirements_file_path = f"{tmpdir}/requirements.txt"

    with open(requirements_file_path, "w+") as req_file:
        for req in requirements:
            req_file.write(req + "\n")

    stored["requirements"] = "requirements.txt"
    return stored


def _save_requirements_file(name, requirements_file_path, stored, tmpdir) -> dict:
    shutil.copyfile(os.path.abspath(requirements_file_path), f"{tmpdir}/requirements.txt")
    stored["requirements"] = requirements_file_path
    return stored


def _upload_metadata(
    meta_data: dict,
    name: str,
    version: str,
    username: str,
    api_key: str,
    project_id: str,
):
    def _get_url(response_content):
        content = json.loads(response_content)
        return content["uploadUrl"]

    json_field = {
        "name": f"{username}/{name}",
        "version": version,
        "metadata": meta_data,
    }
    if project_id:
        json_field["project_id"] = project_id
    response = requests.post(
        LIGHTNING_MODELS_PUBLIC_REGISTRY,
        auth=HTTPBasicAuth(username, api_key),
        json=json_field,
    )
    if response.status_code != 200:
        raise ConnectionRefusedError(f"Unable to upload content.\n Error: {response.content}\n for load: {json_field}")
    return _get_url(response.content)


def _save_meta_data(name, stored, version, model, username, api_key, project_id):
    def _process_stored(stored: dict):
        processed_dict = {}
        for key, val in stored.items():
            if "code" in key:
                for code_key, code_val in stored[key].items():
                    processed_dict[f"stored_code_{code_key}"] = code_val
            else:
                processed_dict[f"stored_{key}"] = val
        return processed_dict

    meta_data = {"cls": model.__class__.__name__}
    meta_data.update(_process_stored(stored))

    return _upload_metadata(
        meta_data,
        name=name,
        version=version,
        username=username,
        api_key=api_key,
        project_id=project_id,
    )


def _submit_data_to_url(url: str, tmpdir: str, progress_bar: bool) -> None:
    def _make_tar(tmpdir, archive_output_path):
        with tarfile.open(archive_output_path, "w:gz") as tar:
            tar.add(tmpdir)

    def upload_from_file(src, dst):
        file_size = os.path.getsize(src)
        with open(src, "rb") as fd:
            with tqdm(
                desc="Uploading",
                total=file_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as t:
                reader_wrapper = CallbackIOWrapper(t.update, fd, "read")
                response = requests.put(dst, data=reader_wrapper)
                response.raise_for_status()

    archive_path = f"{tmpdir}/data.tar.gz"
    _make_tar(tmpdir, archive_path)
    if progress_bar:
        upload_from_file(archive_path, url)
    else:
        requests.put(url, data=open(archive_path, "rb"))


def _download_tarfile(download_url: str, output_dir: str, progress_bar: bool) -> None:
    with requests.get(download_url, stream=True) as req_stream:
        total_size_in_bytes = int(req_stream.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte

        download_progress_bar = None
        if progress_bar:
            download_progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(f"{output_dir}/data.tar.gz", "wb") as f:
            for chunk in req_stream.iter_content(chunk_size=block_size):
                if download_progress_bar:
                    download_progress_bar.update(len(chunk))
                f.write(chunk)
        if download_progress_bar:
            download_progress_bar.close()


def _common_clean_up(output_dir: str) -> None:
    data_file_path = f"{output_dir}/data.tar.gz"
    dir_file_path = f"{output_dir}/extracted"
    if os.path.exists(data_file_path):
        os.remove(data_file_path)
    shutil.rmtree(dir_file_path)


def _download_and_extract_data_to(output_dir: str, download_url: str, progress_bar: bool) -> None:
    try:
        _download_tarfile(download_url, output_dir, progress_bar)

        tar = tarfile.open(f"{output_dir}/data.tar.gz", "r:gz")
        tmpdir_name = tar.getnames()[0]
        tar.extractall(path=f"{output_dir}/extracted")
        tar.close()

        root = f"{output_dir}"
        for filename in os.listdir(os.path.join(root, "extracted", tmpdir_name)):
            abs_file_name = os.path.join(root, "extracted", tmpdir_name, filename)
            func = shutil.copytree if os.path.isdir(abs_file_name) else shutil.copy

            dst_file_name = os.path.join(root, filename)
            if os.path.exists(dst_file_name):
                if os.path.isdir(dst_file_name):
                    shutil.rmtree(dst_file_name)
                else:
                    os.remove(dst_file_name)

            func(abs_file_name, os.path.join(root, filename))

        if not os.path.isdir(f"{output_dir}"):
            raise NotADirectoryError(
                f"Data downloading to the output directory: {output_dir} failed."
                f" Maybe try again or contact the model owner?"
            )
    except Exception as ex:
        _common_clean_up(output_dir)
        raise ex
    else:
        _common_clean_up(output_dir)


def _get_linked_output_dir(src_dir: str):
    # The last sub-folder will be our version
    version_folder_name = PurePath(src_dir).parts[-1]

    if version_folder_name == "latest":
        return str(PurePath(src_dir).parent.joinpath("version_latest"))
    else:
        replaced_ver = version_folder_name.replace(".", "_")
        return str(PurePath(src_dir).parent.joinpath(f"version_{replaced_ver}"))
