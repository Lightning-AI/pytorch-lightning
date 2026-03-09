#!/usr/bin/env python3
"""Simple test script to verify the AdvancedProfiler fix works.

This reproduces the original bug and verifies it's fixed.

"""

import os
import sys
import tempfile

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from lightning.pytorch.profilers.advanced import AdvancedProfiler

    print("✓ Successfully imported AdvancedProfiler")
except ImportError as e:
    print(f"✗ Failed to import AdvancedProfiler: {e}")
    sys.exit(1)


def test_stop_nonexistent_action():
    """Test that stopping a non-existent action doesn't raise ValueError."""
    print("\n=== Testing stop of nonexistent action ===")

    with tempfile.TemporaryDirectory() as tmp_dir:
        profiler = AdvancedProfiler(dirpath=tmp_dir, filename="test")

        try:
            # This should NOT raise ValueError after the fix
            profiler.stop("run_test_evaluation")
            profiler.stop("some_nonexistent_action")
            print("✓ Stopping nonexistent actions completed without error")
            return True
        except ValueError as e:
            print(f"✗ ValueError was still raised: {e}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return False


def test_normal_profiling_still_works():
    """Test that normal profiling functionality still works."""
    print("\n=== Testing normal profiling still works ===")

    with tempfile.TemporaryDirectory() as tmp_dir:
        profiler = AdvancedProfiler(dirpath=tmp_dir, filename="test")

        try:
            # Normal profiling should still work
            with profiler.profile("test_action"):
                sum(range(100))  # Some work

            # Should be able to get summary
            summary = profiler.summary()
            if "test_action" in summary:
                print("✓ Normal profiling works correctly")
                return True
            print("✗ Normal profiling summary doesn't contain expected action")
            return False

        except Exception as e:
            print(f"✗ Normal profiling failed: {e}")
            return False


def test_mixed_usage():
    """Test mixed usage of stop on nonexistent and normal profiling."""
    print("\n=== Testing mixed usage ===")

    with tempfile.TemporaryDirectory() as tmp_dir:
        profiler = AdvancedProfiler(dirpath=tmp_dir, filename="test")

        try:
            # Stop nonexistent action - should not error
            profiler.stop("nonexistent1")

            # Normal profiling
            with profiler.profile("real_action"):
                sum(range(50))

            # Stop another nonexistent action - should not error
            profiler.stop("nonexistent2")

            # Check summary contains the real action
            summary = profiler.summary()
            if "real_action" in summary:
                print("✓ Mixed usage works correctly")
                return True
            print("✗ Mixed usage failed - real action not in summary")
            return False

        except Exception as e:
            print(f"✗ Mixed usage failed: {e}")
            return False


def main():
    """Run all tests."""
    print("Testing AdvancedProfiler fix for issue #9136")
    print("=" * 50)

    tests = [
        test_stop_nonexistent_action,
        test_normal_profiling_still_works,
        test_mixed_usage,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 50)
    print("SUMMARY:")

    all_passed = all(results)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("✓ The fix successfully resolves the issue!")
        return 0
    print("✗ SOME TESTS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
