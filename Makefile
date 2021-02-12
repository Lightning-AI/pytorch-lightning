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
	rm -rf ./docs/source/generated
	rm -rf ./docs/source/*/generated
	rm -rf ./docs/source/api

test: clean
	pip install -r requirements/devel.txt
	# install APEX, see https://github.com/NVIDIA/apex#linux

	# use this to run tests
	python -m coverage run --source pytorch_lightning -m pytest pytorch_lightning tests pl_examples -v --flake8
	python -m coverage report

docs: clean
	pip install --quiet -r requirements/docs.txt
	python -m sphinx -b html -W docs/source docs/build
