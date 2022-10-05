.PHONY: test clean docs

# to imitate SLURM set only single node
export SLURM_LOCALID=0
# assume you have installed need packages
export SPHINX_MOCK_REQUIREMENTS=1
# install only Lightning Trainer packages
export PACKAGE_NAME=pytorch

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
	rm -rf ./docs/source-lit/api
	rm -rf ./docs/source-lit/generated
	rm -rf ./docs/source-lit/*/generated
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	rm -rf src/lightning/*/

test: clean
	# Review the CONTRIBUTING documentation for other ways to test.
	pip install -e . \
	-r requirements/pytorch/base.txt \
	-r requirements/app/base.txt \
	-r requirements/lite/base.txt \
	-r requirements/pytorch/test.txt \
	-r requirements/app/test.txt

	# run tests with coverage
	python -m coverage run --source src/pytorch_lightning -m pytest src/pytorch_lightning tests/tests_pytorch -v
	python -m coverage run --source src/lightning_app -m pytest tests/tests_app -v
	python -m coverage run --source src/lightning_lite -m pytest src/lightning_lite tests/tests_lite -v
	python -m coverage report

docs: clean
	pip install -e . --quiet -r requirements/lit/docs.txt
	cd docs/source-lit && $(MAKE) html

update:
	git submodule update --init --recursive --remote
