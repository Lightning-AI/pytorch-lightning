import warnings

from pytest import PytestWarning

# Ignore cleanup warnings from pytest (rarely happens due to a race condition when executing pytest in parallel)
warnings.filterwarnings("ignore", category=PytestWarning, message=r".*\(rm_rf\) error removing.*")
