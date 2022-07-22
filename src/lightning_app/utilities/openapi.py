import json
from typing import Any, Dict, io

import click
from pathlib import Path

import yaml


def _duplicate_checker(js):
    result = {}
    for name, value in js:
        if name in result:
            raise ValueError('Failed to load JSON: duplicate key {0}.'.format(name))
        result[name] = value
    return result


def string2dict(text):
    if not isinstance(text, str):
        text = text.decode('utf-8')
    try:
        js = json.loads(text, object_pairs_hook=_duplicate_checker)
        return js
    except ValueError as e:
        raise ValueError('Failed to load JSON: {0}.'.format(str(e)))


def is_openapi(obj):
    return hasattr(obj, "swagger_types")


def create_openapi_object(json_obj: Dict, target: Any):
    """ Create the openAPI object from the given json dict and based on the target object
    We use the target object to make new object from the given json spec and hence target
    must be a valid object.
    """
    if not isinstance(json_obj, dict):
        raise TypeError("json_obj must be a dictionary")
    if not is_openapi(target):
        raise TypeError("target must be an openapi object")

    target_attribs = {}
    for key, value in json_obj.items():
        try:
            # user provided key is not a valid key on openapi object
            sub_target = getattr(target, key)
        except AttributeError:
            raise ValueError(f"Field {key} not found in the target object")

        if is_openapi(sub_target):  # it's an openapi object
            target_attribs[key] = create_openapi_object(value, sub_target)
        else:
            target_attribs[key] = value

        # TODO(sherin) - specifically process list and dict and do the validation. Also do the
        #  verification for enum types

    new_target = target.__class__(**target_attribs)
    return new_target
