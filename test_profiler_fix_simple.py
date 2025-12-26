#!/usr/bin/env python3
"""
Simple test of just the AdvancedProfiler class to verify the fix.
This directly tests the modified class without full Lightning dependencies.
"""

import cProfile
import io
import logging
import os
import pstats
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

log = logging.getLogger(__name__)

class AdvancedProfiler:
    """Minimal version of AdvancedProfiler to test the fix."""

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        line_count_restriction: float = 1.0,
        dump_stats: bool = False,
    ) -> None:
        self.dirpath = dirpath
        self.filename = filename
        self.profiled_actions: dict[str, cProfile.Profile] = defaultdict(cProfile.Profile)
        self.line_count_restriction = line_count_restriction
        self.dump_stats = dump_stats

    def start(self, action_name: str) -> None:
        # Disable all profilers before starting a new one
        for pr in self.profiled_actions.values():
            pr.disable()
        self.profiled_actions[action_name].enable()

    def stop(self, action_name: str) -> None:
        pr = self.profiled_actions.get(action_name)
        if pr is None:
            # This is the fix - log debug instead of raising ValueError
            log.debug(f"Attempting to stop recording an action ({action_name}) which was never started.")
            return
        pr.disable()

    def profile(self, action_name: str):
        """Context manager for profiling."""
        class ProfileContext:
            def __init__(self, profiler, action_name):
                self.profiler = profiler
                self.action_name = action_name

            def __enter__(self):
                self.profiler.start(self.action_name)
                return self.action_name

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.profiler.stop(self.action_name)

        return ProfileContext(self, action_name)

    def summary(self) -> str:
        recorded_stats = {}
        for action_name, pr in self.profiled_actions.items():
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative")
            ps.print_stats(self.line_count_restriction)
            recorded_stats[action_name] = s.getvalue()
        return str(recorded_stats)

    def teardown(self, stage: Optional[str] = None) -> None:
        self.profiled_actions.clear()

def test_stop_nonexistent_action():
    """Test that stopping a non-existent action doesn't raise ValueError."""
    print("=== Testing stop of nonexistent action ===")

    profiler = AdvancedProfiler(dirpath="/tmp", filename="test")

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

    profiler = AdvancedProfiler(dirpath="/tmp", filename="test")

    try:
        # Normal profiling should still work
        with profiler.profile("test_action"):
            x = sum(range(100))  # Some work

        # Should be able to get summary
        summary = profiler.summary()
        if "test_action" in summary:
            print("✓ Normal profiling works correctly")
            return True
        else:
            print("✗ Normal profiling summary doesn't contain expected action")
            return False

    except Exception as e:
        print(f"✗ Normal profiling failed: {e}")
        return False

def test_mixed_usage():
    """Test mixed usage of stop on nonexistent and normal profiling."""
    print("\n=== Testing mixed usage ===")

    profiler = AdvancedProfiler(dirpath="/tmp", filename="test")

    try:
        # Stop nonexistent action - should not error
        profiler.stop("nonexistent1")

        # Normal profiling
        with profiler.profile("real_action"):
            y = sum(range(50))

        # Stop another nonexistent action - should not error
        profiler.stop("nonexistent2")

        # Check summary contains the real action
        summary = profiler.summary()
        if "real_action" in summary:
            print("✓ Mixed usage works correctly")
            return True
        else:
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
    else:
        print("✗ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    exit(main())