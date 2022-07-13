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

test: clean
	# Review the CONTRIBUTING documentation for other ways to test.
	pip install -e . -r requirements/pytorch/devel.txt
	pip install -r requirements/pytorch/strategies.txt
	# run tests with coverage
	python -m coverage run --source pytorch_lightning -m pytest pytorch_lightning tests -v
	python -m coverage report

docs: clean
	pip install -e . --quiet -r requirements/pytorch/docs.txt
	python -m sphinx -b html -W --keep-going docs/source-pytorch docs/build

update:
	git submodule update --init --recursive --remote
