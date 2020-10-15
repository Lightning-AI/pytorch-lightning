from typing import List
from pytorch_lightning.callbacks import Callback
import inspect

def create_scriptable_callback(funcs_name: List[str], logic_func: callable) -> Callback:
    r"""
    This function enables to dynamically override any function of the current `pytorch_lightning.callbacks.Callback` class. 
    Can be useful to testing.

    Args:
        funcs_name: List of function name to override
        logic_func: Lambda function which will return a code str to be bounded to the ScriptableCallback
            Note: lambda will receive x, a tuple (func_idx, func_name, args)
    """

    class ScriptableCallback(Callback):
        pass

    ld = {}
    for func_idx, func_name in enumerate(funcs_name):
        func = getattr(Callback, func_name)
        args = ', '.join(inspect.signature(func).parameters.keys())
        sub_func = logic_func((func_idx, func_name, args))
        func_code = f"""def {func_name}({args}):\n  {sub_func}"""
        exec(func_code, None, ld)
    
    for name, value in ld.items():
        setattr(ScriptableCallback, name, value)
    
    return ScriptableCallback()