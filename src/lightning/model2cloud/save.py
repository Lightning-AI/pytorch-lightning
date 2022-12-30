import inspect
import json
import logging
import os
import shutil
import tarfile
from pathlib import PurePath

import requests
import torch
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

from lightning.app.model2cloud.utils import LIGHTNING_CLOUD_URL

logging.basicConfig(level=logging.INFO)


def _check_id(id: str):
    if id[-1] != "/":
        id += "/"
    assert id.count("/") == 2, "The format for the ID should be: <username>/<model_name or any id>"
    return id


def _save_checkpoint(name, checkpoint, tmpdir, stored):
    checkpoint_file_path = f"{tmpdir}/checkpoint.ckpt"
    torch.save(checkpoint, checkpoint_file_path)
    stored["checkpoint"] = "checkpoint.ckpt"
    return stored


def _save_checkpoint_from_path(name, path, tmpdir, stored):
    checkpoint_file_path = path
    shutil.copy(checkpoint_file_path, f"{tmpdir}/checkpoint.ckpt")
    stored["checkpoint"] = "checkpoint.ckpt"
    return stored


def _save_model_weights(name, model_state_dict, tmpdir, stored, *args, **kwargs):
    # For now we assume that it's always going to be public
    weights_file_path = f"{tmpdir}/weights.pt"
    torch.save(model_state_dict, weights_file_path, *args, **kwargs)
    stored["weights"] = "weights.pt"
    return stored


def _save_model(name, model, tmpdir, stored, *args, **kwargs):
    # For now we assume that it's always going to be public
    model_file_path = f"{tmpdir}/model"
    torch.save(model, model_file_path, *args, **kwargs)
    stored["model"] = "model"
    return stored


def _save_model_code(name, model_cls, source_code_path, tmpdir, stored):
    if source_code_path:
        source_code_path = os.path.abspath(source_code_path)
        assert os.path.exists(source_code_path), f"Given path {source_code_path} does not exist."

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
            assert os.path.splitext(source_code_path)[-1] == ".py", (
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
            assert os.path.splitext(model_class_path)[-1] == ".py", (
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


def _save_requirements_file(name, requirements_file_path, stored, tmpdir):
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
        f"{LIGHTNING_CLOUD_URL}/v1/models",
        auth=HTTPBasicAuth(username, api_key),
        json=json_field,
    )

    assert (
        response.status_code == 200
    ), f"Unable to upload content, did you pass correct credentials? Error: {response.content}"
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

    meta_data = {
        "cls": model.__class__.__name__,
    }

    meta_data.update(_process_stored(stored))

    return _upload_metadata(
        meta_data,
        name=name,
        version=version,
        username=username,
        api_key=api_key,
        project_id=project_id,
    )


def _submit_data_to_url(url: str, tmpdir: str, progress_bar: bool):
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


def _download_and_extract_data_to(output_dir: str, download_url: str, progress_bar: bool):
    def _common_clean_up():
        data_file_path = f"{output_dir}/data.tar.gz"
        dir_file_path = f"{output_dir}/extracted"
        if os.path.exists(data_file_path):
            os.remove(data_file_path)
        shutil.rmtree(dir_file_path)

    try:
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

        tar = tarfile.open(f"{output_dir}/data.tar.gz", "r:gz")
        tmpdir_name = tar.getnames()[0]
        tar.extractall(path=f"{output_dir}/extracted")
        tar.close()

        root = f"{output_dir}"
        for filename in os.listdir(os.path.join(root, "extracted", tmpdir_name)):
            abs_file_name = os.path.join(root, "extracted", tmpdir_name, filename)
            if os.path.isdir(abs_file_name):
                func = shutil.copytree
            else:
                func = shutil.copy

            dst_file_name = os.path.join(root, filename)
            if os.path.exists(dst_file_name):
                if os.path.isdir(dst_file_name):
                    shutil.rmtree(dst_file_name)
                else:
                    os.remove(dst_file_name)

            func(
                abs_file_name,
                os.path.join(root, filename),
            )

        assert os.path.isdir(f"{output_dir}"), (
            "Data downloading to the output"
            f" directory: {output_dir} failed. Maybe try again or contact the model owner?"
        )
    except Exception as e:
        _common_clean_up()
        raise e
    else:
        _common_clean_up()


def get_linked_output_dir(src_dir: str):
    # The last sub-folder will be our version
    version_folder_name = PurePath(src_dir).parts[-1]

    if version_folder_name == "latest":
        return str(PurePath(src_dir).parent.joinpath("version_latest"))
    else:
        replaced_ver = version_folder_name.replace(".", "_")
        return str(PurePath(src_dir).parent.joinpath(f"version_{replaced_ver}"))
