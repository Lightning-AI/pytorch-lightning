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

import os
import signal
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

from typing_extensions import TypedDict

from lightning.app.core.work import LightningWork
from lightning.app.storage.drive import Drive
from lightning.app.storage.payload import Payload
from lightning.app.utilities.app_helpers import _collect_child_process_pids, Logger
from lightning.app.utilities.packaging.tarfile import clean_tarfile, extract_tarfile
from lightning.app.utilities.tracer import Tracer

logger = Logger(__name__)


class Code(TypedDict):
    drive: Drive
    name: str


class TracerPythonScript(LightningWork):
    _start_method = "spawn"

    def on_before_run(self):
        """Called before the python script is executed."""

    def on_after_run(self, res: Any):
        """Called after the python script is executed."""
        for name in self.outputs:
            setattr(self, name, Payload(res[name]))

    def configure_tracer(self) -> Tracer:
        """Override this hook to customize your tracer when running PythonScript."""
        return Tracer()

    def __init__(
        self,
        script_path: str,
        script_args: Optional[Union[list, str]] = None,
        outputs: Optional[List[str]] = None,
        env: Optional[Dict] = None,
        code: Optional[Code] = None,
        **kwargs,
    ):
        """The TracerPythonScript class enables to easily run a python script.

        When subclassing this class, you can configure your own :class:`~lightning.app.utilities.tracer.Tracer`
        by :meth:`~lightning.app.components.python.tracer.TracerPythonScript.configure_tracer` method.

        The tracer is quite a magical class. It enables you to inject code into a script execution without changing it.

        Arguments:
            script_path: Path of the python script to run.
            script_path: The arguments to be passed to the script.
            outputs: Collection of object names to collect after the script execution.
            env: Environment variables to be passed to the script.
            kwargs: LightningWork Keyword arguments.

        Raises:
            FileNotFoundError: If the provided `script_path` doesn't exists.

        **How does it work?**

        It works by executing the python script with python built-in `runpy
        <https://docs.python.org/3/library/runpy.html>`_ run_path method.
        This method takes any python globals before executing the script,
        e.g., you can modify classes or function from the script.

        Example:

            >>> from lightning.app.components.python import TracerPythonScript
            >>> f = open("a.py", "w")
            >>> f.write("print('Hello World !')")
            22
            >>> f.close()
            >>> python_script = TracerPythonScript("a.py")
            >>> python_script.run()
            Hello World !
            >>> os.remove("a.py")

        In the example below, we subclass the  :class:`~lightning.app.components.python.TracerPythonScript`
        component and override its configure_tracer method.

        Using the Tracer, we are patching the ``__init__`` method of the PyTorch Lightning Trainer.
        Once the script starts running and if a Trainer is instantiated, the provided ``pre_fn`` is
        called and we inject a Lightning callback.

        This callback has a reference to the work and on every batch end, we are capturing the
        trainer ``global_step`` and ``best_model_path``.

        Even more interesting, this component works for ANY PyTorch Lightning script and
        its state can be used in real time in a UI.

        .. literalinclude:: ../../../examples/app_components/python/component_tracer.py
            :language: python


        Once implemented, this component can easily be integrated within a larger app
        to execute a specific python script.

        .. literalinclude:: ../../../examples/app_components/python/app.py
            :language: python
        """
        super().__init__(**kwargs)
        self.script_path = str(script_path)
        if isinstance(script_args, str):
            script_args = script_args.split(" ")
        self.script_args = script_args if script_args else []
        self.original_args = deepcopy(self.script_args)
        self.env = env
        self.outputs = outputs or []
        for name in self.outputs:
            setattr(self, name, None)
        self.params = None
        self.drive = code.get("drive") if code else None
        self.code_name = code.get("name") if code else None
        self.restart_count = 0

    def run(
        self,
        params: Optional[Dict[str, Any]] = None,
        restart_count: Optional[int] = None,
        code_dir: Optional[str] = ".",
        **kwargs,
    ):
        """
        Arguments:
            params: A dictionary of arguments to be be added to script_args.
            restart_count: Passes an incrementing counter to enable the re-execution of LightningWorks.
            code_dir: A path string determining where the source is extracted, default is current directory.
        """
        if restart_count:
            self.restart_count = restart_count

        if params:
            self.params = params
            self.script_args = self.original_args + [self._to_script_args(k, v) for k, v in params.items()]

        if self.drive:
            assert self.code_name
            if os.path.exists(self.code_name):
                clean_tarfile(self.code_name, "r:gz")

            if self.code_name in self.drive.list():
                self.drive.get(self.code_name)
                extract_tarfile(self.code_name, code_dir, "r:gz")

        prev_cwd = os.getcwd()
        os.chdir(code_dir)

        if not os.path.exists(self.script_path):
            raise FileNotFoundError(f"The provided `script_path` {self.script_path}` wasn't found.")

        kwargs = {k: v.value if isinstance(v, Payload) else v for k, v in kwargs.items()}

        init_globals = globals()
        init_globals.update(kwargs)

        self.on_before_run()
        env_copy = os.environ.copy()
        if self.env:
            os.environ.update(self.env)
        res = self._run_tracer(init_globals)
        os.chdir(prev_cwd)
        os.environ = env_copy
        return self.on_after_run(res)

    def _run_tracer(self, init_globals):
        sys.argv = [self.script_path]
        tracer = self.configure_tracer()
        return tracer.trace(self.script_path, *self.script_args, init_globals=init_globals)

    def on_exit(self):
        for child_pid in _collect_child_process_pids(os.getpid()):
            os.kill(child_pid, signal.SIGTERM)

    @staticmethod
    def _to_script_args(k: str, v: str) -> str:
        return f"{k}={v}"


__all__ = ["TracerPythonScript"]
