import re
from typing import Dict


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
