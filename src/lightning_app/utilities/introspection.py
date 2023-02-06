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

import ast
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from lightning_app import LightningFlow, LightningWork


class LightningVisitor(ast.NodeVisitor):
    """
    Base class for visitor that finds class definitions based on
    class inheritance.
    Derived classes are expected to define class_name and implement
    the analyze_class_def method.
    Attributes
    ----------
    class_name: str
        Name of class to identify, to be defined in subclasses.
    """

    class_name: Optional[str] = None

    def __init__(self):
        self.found: List[Dict[str, Any]] = []

    def analyze_class_def(self, node: ast.ClassDef) -> Dict[str, Any]:
        return {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = []
        for base in node.bases:
            if type(base) == ast.Attribute:
                bases.append(base.attr)
            elif type(base) == ast.Name:
                bases.append(base.id)
        if self.class_name in bases:
            entry = {"name": node.name, "type": self.class_name}
            entry.update(self.analyze_class_def(node))
            self.found.append(entry)


class LightningModuleVisitor(LightningVisitor):
    """
    Finds Lightning modules based on class inheritance.
    Attributes
    ----------
    class_name: Optional[str]
        Name of class to identify.
    methods: Set[str]
        Names of methods that are part of the LightningModule API.
    hooks: Set[str]
        Names of hooks that are part of the LightningModule API.
    """

    class_name: Optional[str] = "LightningModule"

    methods: Set[str] = {
        "configure_optimizers",
        "forward",
        "freeze",
        "log",
        "log_dict",
        "print",
        "save_hyperparameters",
        "test_step",
        "test_step_end",
        "test_epoch_end",
        "to_onnx",
        "to_torchscript",
        "training_step",
        "training_step_end",
        "training_epoch_end",
        "unfreeze",
        "validation_step",
        "validation_step_end",
        "validation_epoch_end",
    }

    hooks: Set[str] = {
        "backward",
        "get_progress_bar_dict",
        "manual_backward",
        "manual_optimizer_step",
        "on_after_backward",
        "on_before_zero_grad",
        "on_fit_start",
        "on_fit_end",
        "on_load_checkpoint",
        "on_save_checkpoint",
        "on_pretrain_routine_start",
        "on_pretrain_routine_end",
        "on_test_batch_start",
        "on_test_batch_end",
        "on_test_epoch_start",
        "on_test_epoch_end",
        "on_train_batch_start",
        "on_train_batch_end",
        "on_train_epoch_start",
        "on_train_epoch_end",
        "on_validation_batch_start",
        "on_validation_batch_end",
        "on_validation_epoch_start",
        "on_validation_epoch_end",
        "optimizer_step",
        "optimizer_zero_grad",
        "prepare_data",
        "setup",
        "tbptt_split_batch",
        "teardown",
        "train_dataloader",
        "val_dataloader",
        "test_dataloader",
        "transfer_batch_to_device",
    }


class LightningDataModuleVisitor(LightningVisitor):
    """
    Finds Lightning data modules based on class inheritance.
    Attributes
    ----------
    class_name: Optional[str]
        Name of class to identify.
    methods: Set[str]
        Names of methods that are part of the LightningDataModule API.
    """

    class_name = "LightningDataModule"

    methods: Set[str] = {
        "prepare_data",
        "setup",
        "train_dataloader",
        "val_dataloader",
        "test_dataloader",
        "transfer_batch_to_device",
    }


class LightningLoggerVisitor(LightningVisitor):
    """
    Finds Lightning loggers based on class inheritance.
    Attributes
    ----------
    class_name: Optional[str]
        Name of class to identify.
    methods: Set[str]
        Names of methods that are part of the Logger API.
    """

    class_name = "Logger"

    methods: Set[str] = {"log_hyperparams", "log_metrics"}


class LightningCallbackVisitor(LightningVisitor):
    """
    Finds Lightning callbacks based on class inheritance.
    Attributes
    ----------
    class_name: Optional[str]
        Name of class to identify.
    methods: Set[str]
        Names of methods that are part of the Logger API.
    """

    class_name = "Callback"

    methods: Set[str] = {
        "setup",
        "teardown",
        "on_init_start",
        "on_init_end",
        "on_fit_start",
        "on_fit_end",
        "on_sanity_check_start",
        "on_sanity_check_end",
        "on_train_batch_start",
        "on_train_batch_end",
        "on_train_epoch_start",
        "on_train_epoch_end",
        "on_validation_epoch_start",
        "on_validation_epoch_end",
        "on_test_epoch_start",
        "on_test_epoch_end",
        "on_epoch_start",
        "on_epoch_end",
        "on_batch_start",
        "on_validation_batch_start",
        "on_validation_batch_end",
        "on_test_batch_start",
        "on_test_batch_end",
        "on_batch_end",
        "on_train_start",
        "on_train_end",
        "on_pretrain_routine_start",
        "on_pretrain_routine_end",
        "on_validation_start",
        "on_validation_end",
        "on_test_start",
        "on_test_end",
        "on_keyboard_interrupt",
        "on_save_checkpoint",
        "on_load_checkpoint",
    }


class LightningStrategyVisitor(LightningVisitor):
    """
    Finds Lightning callbacks based on class inheritance.
    Attributes
    ----------
    class_name: Optional[str]
        Name of class to identify.
    methods: Set[str]
        Names of methods that are part of the Logger API.
    """

    class_name = "Strategy"

    methods: Set[str] = {
        "setup",
        "train",
        "training_step",
        "validation_step",
        "test_step",
        "backward",
        "barrier",
        "broadcast",
        "sync_tensor",
    }


class LightningTrainerVisitor(LightningVisitor):
    class_name = "Trainer"


class LightningCLIVisitor(LightningVisitor):
    class_name = "LightningCLI"


class LightningPrecisionPluginVisitor(LightningVisitor):
    class_name = "PrecisionPlugin"


class LightningAcceleratorVisitor(LightningVisitor):
    class_name = "Accelerator"


class LightningLoopVisitor(LightningVisitor):
    class_name = "Loop"


class TorchMetricVisitor(LightningVisitor):
    class_name = "Metric"


class LightningLiteVisitor(LightningVisitor):  # deprecated
    class_name = "LightningLite"


class FabricVisitor(LightningVisitor):
    class_name = "Fabric"


class LightningProfilerVisitor(LightningVisitor):
    class_name = "Profiler"


class Scanner:
    """
    Finds relevant Lightning objects in files in the file system.
    Attributes
    ----------
    visitor_classes: List[Type]
        List of visitor classes to use when traversing files.
    Parameters
    ----------
    path: str
        Path to file, or directory where to look for files to scan.
    glob_pattern: str
        Glob pattern to use when looking for files in the path,
        applied when path is a directory. Default is "**/*.py".
    """

    # TODO: Finalize introspecting the methods from all the discovered methods.
    visitor_classes: List[Type] = [
        LightningCLIVisitor,
        LightningTrainerVisitor,
        LightningModuleVisitor,
        LightningDataModuleVisitor,
        LightningCallbackVisitor,
        LightningStrategyVisitor,
        LightningPrecisionPluginVisitor,
        LightningAcceleratorVisitor,
        LightningLoggerVisitor,
        LightningLoopVisitor,
        TorchMetricVisitor,
        LightningLiteVisitor,  # deprecated
        FabricVisitor,
        LightningProfilerVisitor,
    ]

    def __init__(self, path: str, glob_pattern: str = "**/*.py"):
        path_ = Path(path)
        if path_.is_dir():
            self.paths = path_.glob(glob_pattern)
        else:
            self.paths = [path_]

        self.modules_found: List[Dict[str, Any]] = []

    def has_class(self, cls) -> bool:
        # This method isn't strong enough as it is using only `ImportFrom`.
        # TODO: Use proper classDef scanning.
        classes = []

        for path in self.paths:
            try:
                module = ast.parse(path.open().read())
            except SyntaxError:
                print(f"Error while parsing {path}: SKIPPING")
                continue

            for node in ast.walk(module):

                if isinstance(node, ast.ImportFrom):
                    for import_from_cls in node.names:
                        classes.append(import_from_cls.name)

                if isinstance(node, ast.Call):
                    cls_name = getattr(node.func, "attr", None)
                    if cls_name:
                        classes.append(cls_name)

        return cls.__name__ in classes

    def scan(self) -> List[Dict[str, str]]:
        """
        Finds Lightning modules in files, returning importable
        objects.
        Returns
        -------
        List[Dict[str, Any]]
            List of dicts containing all metadata required
            to import modules found.
        """
        modules_found: Dict[str, List[Dict[str, Any]]] = {}

        for path in self.paths:
            try:
                module = ast.parse(path.open().read())
            except SyntaxError:
                print(f"Error while parsing {path}: SKIPPING")
                continue
            for visitor_class in self.visitor_classes:
                visitor = visitor_class()
                visitor.visit(module)
                if not visitor.found:
                    continue
                _path = str(path)
                ns_info = {
                    "file": _path,
                    "namespace": _path.replace("/", ".").replace(".py", ""),
                }
                modules_found[visitor_class.class_name] = [{**entry, **ns_info} for entry in visitor.found]

        return modules_found


def _is_method_context(component: Union["LightningFlow", "LightningWork"], selected_caller_name: str) -> bool:
    """Checks whether the call to a component originates from within the context of the component's ``__init__``
    method."""
    frame = inspect.currentframe().f_back

    while frame is not None:
        caller_name = frame.f_code.co_name
        caller_self = frame.f_locals.get("self")
        if caller_name == selected_caller_name and caller_self is component:
            # the call originates from a frame under component.__init__
            return True
        frame = frame.f_back

    return False


def _is_init_context(component: Union["LightningFlow", "LightningWork"]) -> bool:
    """Checks whether the call to a component originates from within the context of the component's ``__init__``
    method."""
    return _is_method_context(component, "__init__")


def _is_run_context(component: Union["LightningFlow", "LightningWork"]) -> bool:
    """Checks whether the call to a component originates from within the context of the component's ``run``
    method."""
    return _is_method_context(component, "run") or _is_method_context(component, "load_state_dict")
