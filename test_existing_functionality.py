#!/usr/bin/env python3
"""
Test to verify existing AdvancedProfiler functionality still works after the fix.
Based on existing tests from the test suite.
"""

import tempfile
from pathlib import Path
import time

# Import our modified AdvancedProfiler
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from lightning.pytorch.profilers.advanced import AdvancedProfiler


def test_advanced_profiler_deepcopy():
    """Test that AdvancedProfiler can be deep copied."""
    from copy import deepcopy

    with tempfile.TemporaryDirectory() as tmp_path:
        profiler = AdvancedProfiler(dirpath=tmp_path, filename="profiler")
        profiler.describe()
        try:
            result = deepcopy(profiler)
            print("✓ AdvancedProfiler deepcopy works")
            return True
        except Exception as e:
            print(f"✗ AdvancedProfiler deepcopy failed: {e}")
            return False


def test_advanced_profiler_nested():
    """Test that AdvancedProfiler handles nested profiling actions."""
    with tempfile.TemporaryDirectory() as tmp_path:
        profiler = AdvancedProfiler(dirpath=tmp_path, filename="profiler")

        try:
            with profiler.profile("outer"), profiler.profile("inner"):
                pass  # Should not raise ValueError
            print("✓ AdvancedProfiler nested profiling works")
            return True
        except Exception as e:
            print(f"✗ AdvancedProfiler nested profiling failed: {e}")
            return False


def test_advanced_profiler_basic_functionality():
    """Test basic profiling functionality."""
    with tempfile.TemporaryDirectory() as tmp_path:
        profiler = AdvancedProfiler(dirpath=tmp_path, filename="profiler")

        try:
            # Test basic profiling
            with profiler.profile("test_action"):
                time.sleep(0.01)  # Small delay to register some activity

            # Test that we can get a summary
            summary = profiler.summary()
            if "test_action" not in summary:
                print("✗ test_action not found in profiler summary")
                return False

            print("✓ AdvancedProfiler basic functionality works")
            return True
        except Exception as e:
            print(f"✗ AdvancedProfiler basic functionality failed: {e}")
            return False


def test_advanced_profiler_stop_started_action():
    """Test that stopping a properly started action still works."""
    with tempfile.TemporaryDirectory() as tmp_path:
        profiler = AdvancedProfiler(dirpath=tmp_path, filename="profiler")

        try:
            # Start an action
            profiler.start("test_action")

            # Do some work
            time.sleep(0.01)

            # Stop the action - this should work
            profiler.stop("test_action")

            # Verify it's in the summary
            summary = profiler.summary()
            if "test_action" not in summary:
                print("✗ Properly started/stopped action not found in summary")
                return False

            print("✓ AdvancedProfiler start/stop of real action works")
            return True
        except Exception as e:
            print(f"✗ AdvancedProfiler start/stop of real action failed: {e}")
            return False


def test_original_bug_scenario():
    """Test the original bug scenario is now fixed."""
    with tempfile.TemporaryDirectory() as tmp_path:
        profiler = AdvancedProfiler(dirpath=tmp_path, filename="profiler")

        try:
            # Simulate the problematic scenario: stop an action that was never started
            # This specifically mimics the "run_test_evaluation" error from the issue
            profiler.stop("run_test_evaluation")
            profiler.stop("run_validation_evaluation")
            profiler.stop("some_random_action")

            # Verify profiler is still functional after these calls
            with profiler.profile("after_fix_test"):
                time.sleep(0.01)

            summary = profiler.summary()
            if "after_fix_test" not in summary:
                print("✗ Profiler not functional after stopping nonexistent actions")
                return False

            print("✓ Original bug scenario is fixed")
            return True
        except ValueError as e:
            print(f"✗ Original bug still exists - ValueError raised: {e}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error in bug fix test: {e}")
            return False


def main():
    """Run all tests."""
    print("Testing AdvancedProfiler - verifying fix and existing functionality")
    print("=" * 70)

    tests = [
        ("Deepcopy functionality", test_advanced_profiler_deepcopy),
        ("Nested profiling", test_advanced_profiler_nested),
        ("Basic functionality", test_advanced_profiler_basic_functionality),
        ("Start/stop real actions", test_advanced_profiler_stop_started_action),
        ("Original bug fix", test_original_bug_scenario),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        results.append(test_func())

    print("\n" + "=" * 70)
    print("FINAL RESULTS:")

    passed = sum(results)
    total = len(results)

    for i, (test_name, _) in enumerate(tests):
        status = "✓ PASS" if results[i] else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nSUMMARY: {passed}/{total} tests passed")

    if passed == total:
        print("✓ ALL TESTS PASSED - Fix is working correctly!")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    exit(main())