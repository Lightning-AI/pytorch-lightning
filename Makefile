.PHONY: test clean docs

# to imitate SLURM set only single node
export SLURM_LOCALID=0
# assume you have installed need packages
export SPHINX_MOCK_REQUIREMENTS=1

clean:
	# clean all temp runs
	rm -rf $(shell find . -name "mlruns")
	rm -rf $(shell find . -name "lightning_log")
	rm -rf $(shell find . -name "lightning_logs")
	rm -rf _ckpt_*
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf ./docs/build
	rm -rf ./docs/source-pytorch/notebooks
	rm -rf ./docs/source-pytorch/generated
	rm -rf ./docs/source-pytorch/*/generated
	rm -rf ./docs/source-pytorch/api
	rm -rf ./docs/source-app/generated
	rm -rf ./docs/source-app/*/generated
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	rm -rf src/lightning/*/

test: test-pytorch

test-pytorch: clean
	# Review the CONTRIBUTING documentation for other ways to test.
	export PACKAGE_NAME=pytorch ; \
	export FREEZE_REQUIREMENTS=1 ; \
	pip install -e .[strategies] -r requirements/pytorch/devel.txt
	# run tests with coverage
	cd tests ; python -m coverage run --source pytorch_lightning -m pytest tests_pytorch -v
	cd tests ; python -m coverage report

docs: docs-pytorch

docs-pytorch: clean update
	export PACKAGE_NAME=pytorch ; \
	export FREEZE_REQUIREMENTS=1 ; \
	pip install -e . --quiet -r requirements/pytorch/docs.txt
	export PL_FAST_DOCS_DEV=1 ; \
	python -m sphinx -b html -W --keep-going docs/source-pytorch docs/build

update:
	git submodule update --init --recursive --remote
