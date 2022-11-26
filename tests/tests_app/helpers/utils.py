import re
from contextlib import contextmanager
from typing import Optional, Type

import pytest


@contextmanager
def no_warning_call(expected_warning: Type[Warning] = UserWarning, match: Optional[str] = None):
    # TODO: Replace with `lightning_utilities.test.warning.no_warning_call`
    # https://github.com/Lightning-AI/utilities/issues/57

    with pytest.warns(None) as record:
        yield

    if match is None:
        try:
            w = record.pop(expected_warning)
        except AssertionError:
            # no warning raised
            return
    else:
        for w in record.list:
            if w.category is expected_warning and re.compile(match).search(w.message.args[0]):
                break
        else:
            return

    msg = "A warning" if expected_warning is None else f"`{expected_warning.__name__}`"
    raise AssertionError(f"{msg} was raised: {w}")
