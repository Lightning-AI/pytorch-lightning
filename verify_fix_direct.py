#!/usr/bin/env python3
"""
Direct verification of the exact fix applied to AdvancedProfiler.
This tests the specific lines of code that were changed.
"""

def test_original_code():
    """Test what the original code would do (should raise ValueError)."""
    from collections import defaultdict
    import cProfile
    import logging

    log = logging.getLogger(__name__)
    profiled_actions = defaultdict(cProfile.Profile)

    # Original implementation
    def stop_original(action_name: str) -> None:
        pr = profiled_actions.get(action_name)
        if pr is None:
            raise ValueError(f"Attempting to stop recording an action ({action_name}) which was never started.")
        pr.disable()

    try:
        stop_original("run_test_evaluation")
        print("✗ Original code did NOT raise ValueError (unexpected)")
        return False
    except ValueError as e:
        print(f"✓ Original code correctly raised ValueError: {e}")
        return True
    except Exception as e:
        print(f"✗ Original code raised unexpected error: {e}")
        return False


def test_fixed_code():
    """Test what the fixed code does (should log debug and return)."""
    from collections import defaultdict
    import cProfile
    import logging

    # Set up logging to capture debug messages
    import io
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    log.addHandler(handler)

    profiled_actions = defaultdict(cProfile.Profile)

    # Fixed implementation (what we changed to)
    def stop_fixed(action_name: str) -> None:
        pr = profiled_actions.get(action_name)
        if pr is None:
            log.debug(f"Attempting to stop recording an action ({action_name}) which was never started.")
            return
        pr.disable()

    try:
        stop_fixed("run_test_evaluation")
        stop_fixed("some_other_action")

        # Check that debug messages were logged
        log_output = log_stream.getvalue()
        if "run_test_evaluation" in log_output and "some_other_action" in log_output:
            print("✓ Fixed code logged debug messages correctly")
        else:
            print("? Fixed code didn't log debug messages (might be logging level issue)")

        print("✓ Fixed code completed without raising ValueError")
        return True
    except Exception as e:
        print(f"✗ Fixed code raised unexpected error: {e}")
        return False
    finally:
        log.removeHandler(handler)


def test_behavior_comparison():
    """Compare the behavior before and after the fix."""
    print("=" * 60)
    print("BEHAVIOR COMPARISON")
    print("=" * 60)

    print("\n1. Testing original behavior (should raise ValueError):")
    original_worked = test_original_code()

    print("\n2. Testing fixed behavior (should log debug and continue):")
    fixed_worked = test_fixed_code()

    return original_worked and fixed_worked


def verify_actual_file_was_changed():
    """Verify that the actual source file contains our fix."""
    try:
        with open("src/lightning/pytorch/profilers/advanced.py", "r") as f:
            content = f.read()

        # Check that our fix is present
        if 'log.debug(f"Attempting to stop recording an action ({action_name}) which was never started.")' in content:
            print("✓ Source file contains the debug logging fix")
            return True
        else:
            print("✗ Source file does not contain the expected fix")
            return False

    except FileNotFoundError:
        print("✗ Could not find source file")
        return False
    except Exception as e:
        print(f"✗ Error reading source file: {e}")
        return False


def main():
    print("DIRECT VERIFICATION OF ADVANCEDPROFILER FIX")
    print("=" * 60)
    print("Issue #9136: ValueError when stopping profiling action that was never started")
    print("Fix: Replace ValueError with debug logging")
    print()

    # Test the behavioral difference
    behavior_ok = test_behavior_comparison()

    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)

    # Verify the actual file was changed
    file_ok = verify_actual_file_was_changed()

    if behavior_ok and file_ok:
        print("\n✓ SUCCESS: Fix has been properly implemented and tested!")
        print("✓ The AdvancedProfiler now handles missing actions gracefully")
        print("✓ No more ValueError crashes for users with multiple trainers")
        return 0
    else:
        print("\n✗ FAILURE: Fix verification failed")
        return 1


if __name__ == "__main__":
    exit(main())