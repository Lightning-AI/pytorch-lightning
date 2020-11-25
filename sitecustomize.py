# see https://docs.python.org/3/library/site.html#module-site for details
import os

if os.environ.get('PL_RUNNING_SPECIAL_TESTS', '0') == '1':
    os.environ['COVERAGE_PROCESS_START'] = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'setup.cfg')

    import coverage
    coverage.process_startup()