# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
import re
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from typing import Optional


@contextmanager
def no_warning_call(expected_warning: type[Warning] = Warning, match: Optional[str] = None) -> Generator:
    """Check that no warning was raised/emitted under this context manager."""
    with warnings.catch_warnings(record=True) as record:
        yield

    for w in record:
        if issubclass(w.category, expected_warning) and (match is None or re.compile(match).search(str(w.message))):
            raise AssertionError(f"`{expected_warning.__name__}` was raised: {w.message!r}")
