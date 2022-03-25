"""This file is called from the hpu-tests.yml pipeline.

The following script run the hpu tests in parallel.
Tests run are:
1. test_inference_only is run on four cards
2. test_all_stages on two cards
3. complete hpu tests using one card
4. complete hpu tests using eight cards.
"""
import itertools
import subprocess
import sys

HPU_TESTS_DICTIONARY = {
    "hpu1_test": "python -m coverage run --source pytorch_lightning -m pytest -sv tests/accelerators/test_hpu.py \
            --forked \
            --junitxml=hpu1_test-results.xml",
    "hpu2_test": "python -m coverage run --source pytorch_lightning -m pytest -sv tests/accelerators/test_hpu.py \
            -k test_all_stages \
            --hpus 2 \
            --verbose \
            --capture=no \
            --forked \
            --junitxml=hpu2_test-results.xml",
    "hpu4_test": "python -m coverage run --source pytorch_lightning -m pytest -sv tests/accelerators/test_hpu.py \
            -k test_inference_only \
            --hpus 4 \
            --capture=no \
            --verbose \
            --forked \
            --junitxml=hpu4_test-results.xml",
    "hpu8_test": "python -m coverage run --source pytorch_lightning -m pytest -sv tests/accelerators/test_hpu.py \
            --forked \
            --hpus 8 \
            --junitxml=hpu8_test-results.xml",
    "hpu1_precision_test": "python -m coverage run --source pytorch_lightning -m pytest -sv tests/plugins/precision/hpu/test_hpu.py \
            --hmp-bf16 'tests/plugins/precision/hpu/ops_bf16.txt' \
            --hmp-fp32 'tests/plugins/precision/hpu/ops_fp32.txt' \
            --forked \
            --junitxml=hpu1_precision_test-results.xml",
}

HPU1_TEST = HPU_TESTS_DICTIONARY["hpu1_test"]
HPU2_TEST = HPU_TESTS_DICTIONARY["hpu2_test"]
HPU4_TEST = HPU_TESTS_DICTIONARY["hpu4_test"]
HPU8_TEST = HPU_TESTS_DICTIONARY["hpu8_test"]
HPU1_PRECISION_TEST = HPU_TESTS_DICTIONARY["hpu1_precision_test"]

PARALLEL_HPU_TESTS_EXECUTION = [[HPU4_TEST, HPU1_TEST], [HPU2_TEST, HPU1_TEST], [HPU8_TEST], [HPU1_PRECISION_TEST]]
TIMEOUT = 60  # seconds
TIMEOUT_EXIT_CODE = -9


def run_hpu_tests_parallel(timeout=TIMEOUT):
    """This function is called to run the HPU tests in parallel.

    We run the tests in sub process to utilize all the eight cards available in the DL1 instance
    Considering the max time taken to run the HPU tests as 60 seconds, we kill the process if the time taken exceeds.

    Args:
        timeout: The threshold time to run the HPU tests in parallel.
            An exception is logged if the threshold timeout gets expired.
            TIMEOUT_EXIT_CODE will be returned as -9 in case of timeout,
            0 in case of success and 4 in case of failure.

    Return:
        The list of exit status of the HPU tests that were run in the subprocess.
        Here, the exit_status 0 means the test run is successful. exit_status 1 means the test run is failed.
    """
    exit_status = []
    with open("stdout_log.txt", "w") as stdout_log, open("error_log.txt", "w") as error_log:
        for hpu_tests in PARALLEL_HPU_TESTS_EXECUTION:
            process_list = [
                subprocess.Popen(
                    each_hpu_test, shell=True, stdout=stdout_log, stderr=error_log, universal_newlines=True
                )
                for each_hpu_test in hpu_tests
            ]
            for process in process_list:
                try:
                    exit_status.append(process.wait(timeout=TIMEOUT))
                except subprocess.TimeoutExpired as e:
                    print(e)
                    print("Killing the process....")
                    process.kill()
                    exit_status.append(TIMEOUT_EXIT_CODE)
    return exit_status


def zip_cmd_exitcode(exit_status):
    """This function is called to zip the tests that were executed with the exit status of the test.

    Args:
        exit_status: The returned exit_status after executing run_hpu_tests_parallel().

    Return:
        A list of hpu tests called and their exit status.
    """
    status_list = []
    status_list = list(zip(list(itertools.chain(*PARALLEL_HPU_TESTS_EXECUTION)), exit_status))
    return status_list


def print_logs(filename):
    """This function is called to read the file and print the logs.

    Args:
        filename: Provide the log filename that need to be print on the console.
    """
    with open(filename) as f:
        print(f.read())


def print_subprocess_logs_and_return_status(exit_status):
    """This function is called to print the logs of subprocess stdout and stderror and return the status of test
    execution.

    Args:
        exit_status: The returned exit_status after executing run_hpu_tests_parallel().

    Return:
        Based on the exit status of the HPU tests, we return success or failure to the main method.
    """
    if all(v == 0 for v in exit_status):
        print("All HPU tests passed")
        file_name = "stdout_log.txt"
        print_logs(file_name)
        return 0
    else:
        print("HPU tests are failing")
        print("Printing stdout_log.txt...")
        file_name = "stdout_log.txt"
        print_logs(file_name)
        print("Printing error_log.txt...")
        file_name = "error_log.txt"
        print_logs(file_name)
        return 1


def main():
    exit_status = run_hpu_tests_parallel(timeout=TIMEOUT)
    status_list = zip_cmd_exitcode(exit_status)
    print("HPU Tests executed and their exit status:", status_list)
    return print_subprocess_logs_and_return_status(exit_status)


if __name__ == "__main__":
    sys.exit(main())
