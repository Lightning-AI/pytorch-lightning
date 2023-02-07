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

import functools
import inspect
import runpy
import sys
import time
from pathlib import Path
from typing import Optional


def get_default_args(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def wrap_fn(fn, cls, method_name, trace, stack_level=1, pre_fn=None, post_fn=None, is_class_method=None):
    """Wrap a function so that its execution can be traced and its args and return values modified."""
    class_name = cls.__qualname__

    @functools.wraps(fn)
    def fn_with_tracing(self, *args, **kwargs):
        if class_name not in trace:
            trace[class_name] = {}

        self_id = id(self)
        stack = inspect.stack()
        frame = stack[stack_level]
        frame_id = id(frame)
        stack_len = len(stack) - 1

        if self_id not in trace[class_name]:
            trace[class_name][self_id] = {}

        if method_name not in trace[class_name][self_id]:
            trace[class_name][self_id][method_name] = {}

        if frame_id not in trace[class_name][self_id][method_name]:
            trace[class_name][self_id][method_name][frame_id] = {}

        trace_entry = trace[class_name][self_id][method_name][frame_id]

        if pre_fn:
            # If a pre_fn is specified, it can both record information
            # in a trace, as well as return modified args and kwargs
            # that will be provided to the actual fn being wrappped
            pre_trace, args, kwargs = pre_fn(self, *args, **kwargs)
            trace_entry["pre"] = pre_trace

        # We record the invocation and the calling location in the trace
        trace_entry["frame"] = {
            "filename": frame.filename,
            "lineno": frame.lineno,
            "function": frame.function,
            "depth": stack_len,
        }

        # we cache the dfeault parameters used during the function call
        trace_entry["default_args"] = get_default_args(fn)

        # we cache also the parameters used during the function call
        trace_entry["call_args"] = kwargs

        trace_entry["call"] = {"start": time.time_ns()}

        if not is_class_method:
            ret = fn(self, *args, **kwargs)
        else:
            ret = fn(*args, **kwargs)

        trace_entry["call"]["end"] = time.time_ns()

        if post_fn:
            # If a post_fn is specified, it can both record information
            # in a trace, as well as modify the value returned from fn
            post_trace, ret = post_fn(self, ret)
            trace_entry["post"] = post_trace

        return ret

    return fn_with_tracing


class Tracer:
    def __init__(self):
        self.methods = []
        self.orig = {}
        self.res = {}

    def add_traced(self, cls, method_name, stack_level=1, pre_fn=None, post_fn=None):
        """Record the fact that we will want to trace method_name in class cls.

        Optionally provide two functions that will execute prior to and after the method. The functions also have a
        chance to modify the input arguments and the return values of the methods.
        """
        self.methods.append((cls, method_name, stack_level, pre_fn, post_fn))

    def _instrument(self):
        """Modify classes by wrapping methods that need to be traced.

        Initialize the output trace dict.
        """
        self.res = {}
        for cls, method, stack_level, pre_fn, post_fn in self.methods:
            fn = getattr(cls, method)
            # this checks if the passed function is a class method
            fn_is_class_method: bool = hasattr(fn, "__self__")

            if cls not in self.orig:
                self.orig[cls] = {}
            self.orig[cls][method] = fn
            wrapped_fn = wrap_fn(
                fn,
                cls,
                method,
                self.res,
                stack_level=stack_level,
                pre_fn=pre_fn,
                post_fn=post_fn,
                is_class_method=fn_is_class_method,
            )

            # this is needed to wrap class methods
            if fn_is_class_method:
                wrapped_fn = classmethod(wrapped_fn)

            setattr(cls, method, wrapped_fn)

    def _restore(self):
        """Restore original methods so classes go back to their initial state."""
        for cls in self.orig:
            for method in self.orig[cls]:
                setattr(cls, method, self.orig[cls][method])

    def _cleanup(self):
        """Cleanup trace by converting trace[class_name][instance_id][method_name][frame_id] to
        trace[class_name][][method_name][] thereby removing references to instance ids."""
        out = {}
        for class_name in self.res:
            out[class_name] = []
            for self_id in self.res[class_name]:
                instance = self.res[class_name][self_id]
                out_instance = {"id": self_id}
                for method_name, method in instance.items():
                    frames = []
                    for frame_id, frame in method.items():
                        frame["id"] = frame_id
                        frames.append(frame)
                    out_instance[method_name] = frames
                out[class_name].append(out_instance)
        self.res = out

    def trace(self, *args, init_globals=None) -> Optional[dict]:
        """Execute the command-line arguments in args after instrumenting for tracing.

        Restore the classes to their initial state after tracing.
        """
        args = list(args)
        script = args[0]
        script_dir = Path(script).parent.absolute()

        sys_path = sys.path[:]
        sys_argv = sys.argv[:]

        sys.path.append(str(script_dir))

        sys.argv = args

        self._instrument()

        res = runpy.run_path(script, run_name="__main__", init_globals=init_globals or globals())

        self._restore()
        self._cleanup()

        sys.path = sys_path[:]
        sys.argv = sys_argv[:]

        res["tracer_res"] = self.res

        return res
