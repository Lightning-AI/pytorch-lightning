import logging
import os
import platform
from typing import List, Union

from packaging.version import Version

import lightning_app

_PACKAGE_REGISTRY_COMMANDS = {
    "quick-start": [
        "curl https://gist.githubusercontent.com/tchaton/b81c8d8ba0f4dd39a47bfa607d81d6d5/raw/a5f84a40c03e349f659e219cc328ffec1b22b2c9/train_script.py > train_script.py",  # noqa E501
        "curl https://gist.githubusercontent.com/tchaton/2df61d77a0adbd0f105b1c2dc01ae83a/raw/f5a86d9e0d05d391dec58545c0c31b43271a3541/requirements.txt > requirements.txt",  # noqa E501
    ]
}

logger = logging.getLogger(__name__)

_PYTHON_GREATER_EQUAL_3_8_0 = Version(platform.python_version()) >= Version("3.8.0")
_LIGHTNING_ENTRYPOINT = "lightning_app.external_components"


def _ensure_package_exists(package_path):
    package_init_file = os.path.join(package_path, "__init__.py")
    if not os.path.exists(package_path):
        os.mkdir(package_path)
    if not os.path.isfile(package_init_file):
        open(package_init_file, mode="a").close()


def _import_external_component_classes(
    external_package_name: str,
    external_classes: List[Union[lightning_app.LightningFlow, lightning_app.LightningWork]],
    validate_external_classes: bool = True,
):
    """Imports a list of external components either a LightningFlow or LightningWork.

    - How it works ?
    - For each component that's not already installed, write an import line in the __init__.py of the package
    """
    from lightning_app import _PROJECT_ROOT, LightningFlow, LightningWork

    external_package_path_parts = [_PROJECT_ROOT, "lightning", "components"]

    for sub_module in external_package_name.split("."):
        external_package_path_parts.append(sub_module)
        _ensure_package_exists(os.path.join(*external_package_path_parts))

    external_package_path_parts.append("__init__.py")
    external_components_import_file = os.path.join(*external_package_path_parts)

    new_imports = []

    with open(external_components_import_file) as f:
        existing_imports = set(f.readlines())

    for external_cls in external_classes:
        if issubclass(external_cls, (LightningWork, LightningFlow)):
            import_line_str = f"from {external_cls.__module__} import {external_cls.__name__} # noqa E511 \n"
            if import_line_str not in existing_imports:
                new_imports.append(import_line_str)
        elif validate_external_classes:
            raise Exception(
                f"Cannot import external component {external_cls.__name__} from {external_cls.__module__}. "
                f"The provided external class isn't a LightningWork or LightningFlow."
            )

    with open(external_components_import_file, "a") as fw:
        fw.writelines(new_imports)

    # TODO: import the classes to make sure it works


def register_all_external_components():

    if _PYTHON_GREATER_EQUAL_3_8_0:
        from importlib.metadata import entry_points

        lightning_entry_points = entry_points().get(_LIGHTNING_ENTRYPOINT, ())
    else:
        from pkg_resources import iter_entry_points

        lightning_entry_points = iter_entry_points(_LIGHTNING_ENTRYPOINT)

    for entrypoint in lightning_entry_points:
        try:
            external_classes = entrypoint.load()()
            _import_external_component_classes(entrypoint.name, external_classes, validate_external_classes=False)
        except Exception as e:
            logger.debug(f"Cannot register entrypoint: {entrypoint.name}, group: {entrypoint.group} error: {str(e)}")


def _pip_uninstall_component_package(component_package: str):
    import subprocess
    import sys

    p = subprocess.Popen(
        [sys.executable, "-m", "pip", "uninstall", component_package, "-y"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _, stderr = p.communicate(timeout=60)  # 30 seconds timeout
    except subprocess.TimeoutExpired:
        p.kill()
        logger.debug(f"Timeout, did not uninstall {component_package}")
    if p.returncode != 0:
        logger.debug(f"Did not not uninstall {component_package}, got this error: {stderr}")


def _pip_install_component_package(component_package: str, force_reinstall=False):
    """pip install a `component_package` and extract the package info from the installation logs."""
    import subprocess
    import sys

    args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        component_package,
        "-v",
    ]

    if force_reinstall:
        args.append("--force-reinstall")

    p = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = p.communicate(timeout=60 * 5)  # 5 minutes timeout
    except subprocess.TimeoutExpired:
        p.kill()
        raise Exception(f"Could not install {component_package}, installation timeout.")

    return

    # stdout = stdout.decode("utf-8")
    # stderr = stderr.decode("utf-8")

    # if p.returncode != 0:
    #     raise Exception(f"Could not pip install {component_package}, got this error: {stderr}.")
    # if not stdout:
    #     raise Exception("Could extract package name from installation logs. No logs found.")

    # component_package_info_str = ""
    # for line in stdout.splitlines() + stderr.splitlines():
    #     if "Installed lightning component package:" in line:
    #         component_package_info_str = line.replace("Installed lightning component package:", "").strip()
    #         break

    # if not component_package_info_str:
    #     # fixme (manskx): need to find a way to get package name from the installation
    #     logger.info("Could not extract lightning component info from installation logs. ")
    #     return None

    # try:
    #     component_package_info_dict = json.loads(component_package_info_str)
    #     assert component_package_info_dict["package"]
    #     assert component_package_info_dict["version"]
    #     assert component_package_info_dict["entry_point"]
    # except Exception:
    #     _pip_uninstall_component_package(component_package)
    #     raise Exception(
    #         "Could not extract lightning component info from installation logs. Cannot parse component package info."
    #     )

    # last_logline = stdout.splitlines()[-1]
    # if "Requirement already satisfied" in last_logline:
    #     warnings.warn(
    #         f"The package {component_package} seems to be already installed but we extracted this "
    #         f"information {component_package_info_str}. "
    #         f"If this is not correct, please uninstall the package and try again. "
    #     )
    # elif "Successfully installed" not in last_logline:
    #     # If the installation is successful, the last log line is something like:
    #     #    Successfully installed <package_name>-<version>
    #     warnings.warn(
    #         f"Lightning is not sure that the package {component_package} is correctly installed, "
    #         f"but we extracted this information {component_package_info_str}. "
    #         f"If this is not correct, please uninstall the package and try again. "
    #     )

    # return component_package_info_dict


def _extract_public_package_name_from_entrypoint(entrypoint):
    """The syntax for entry points is specified as follows:

    "entrypoint.name = entrypoint.value"
    "<name> = [<package>.[<subpackage>.]]<module>[:<object>.<object>]"
    This fun return the <package> part
    """
    return entrypoint.value.split(":")[0].strip().split(".")[0]


def install_external_component(component_package: str):
    """Installs an external lightning component and make it avaiable for usage. `component` param can be a name of
    python package to be installed from pypi, a zip file of a package or a githib url of the package.

    How it works?
     - Run "pip install <component>"
     - Get entry points for `lightning_app.external_components`
     - Register the components from the installed package
    """

    _pip_install_component_package(component_package)

    if component_package not in _PACKAGE_REGISTRY_COMMANDS:
        return

    for command in _PACKAGE_REGISTRY_COMMANDS[component_package]:
        os.system(command)

    # if not installed_component_package_info_dict:
    #     # fixme (manskx): install command should not register all entrypoints
    #     register_all_external_components()
    #     return

    # if _PYTHON_GREATER_EQUAL_3_8_0:
    #     from importlib.metadata import entry_points

    #     lightning_entry_points = entry_points().get(_LIGHTNING_ENTRYPOINT, ())
    # else:
    #     from pkg_resources import iter_entry_points

    #     lightning_entry_points = iter_entry_points(_LIGHTNING_ENTRYPOINT)

    # component_entry_points = [
    #     e
    #     for e in lightning_entry_points
    #     if _extract_public_package_name_from_entrypoint(e) == installed_component_package_info_dict["package"]
    #     and e.name == installed_component_package_info_dict["entry_point"]
    # ]

    # if not component_entry_points:
    #     _pip_uninstall_component_package(installed_component_package_info_dict["package"])
    #     other_entrypoints = "- ".join([f"name: {e.name}, value: {e.value}" for e in lightning_entry_points])

    #     raise Exception(
    #         f"Could not find and entry point for package {installed_component_package_info_dict['package']}, "
    #         f"Make sure that the package is registered to 'lightning_app.external_components' "
    #         f"with the same package name. We found another ({len(lightning_entry_points)}) entrypoints "
    #         f"{other_entrypoints}"
    #     )

    # for entrypoint in component_entry_points:

    #     try:
    #         external_classes = entrypoint.load()()
    #     except Exception as e:
    #         _pip_uninstall_component_package(installed_component_package_info_dict["package"])
    #         raise Exception(
    #             f"Cannot register entrypoint: {entrypoint.name}, value: {entrypoint.value} "
    #             f"group: {entrypoint.group} error: {str(e)}"
    #         )

    #     _import_external_component_classes(entrypoint.name, external_classes)
