#!/bin/bash
set -e

case "$1" in
  base)
    # Run existing tests and explicitly deselect any new tests added by this patch
    pytest -q tests/tests_pytorch/utilities/test_enums.py \
      --deselect=tests/tests_pytorch/utilities/test_enums.py::test_gradient_clip_algorithms_contains_norm_and_value
    ;;
  new)
    # Run newly added tests that should fail before the feature implementation
    pytest -q tests/tests_pytorch/event_logging \
      && pytest -q tests/tests_pytorch/utilities/test_enums.py::test_gradient_clip_algorithms_contains_norm_and_value
    ;;
  *)
    echo "Usage: ./test.sh {base|new}"
    exit 1
    ;;
esac
