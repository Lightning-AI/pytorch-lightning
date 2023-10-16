.PHONY: test clean docs

# to imitate SLURM set only single node
export SLURM_LOCALID=0
# assume you have installed need packages
export SPHINX_MOCK_REQUIREMENTS=0

test:
	pip install -q -r requirements/cli.txt -r requirements/_tests.txt

	# use this to run tests
	rm -rf _ckpt_*
	rm -rf ./lightning_logs
	python -m coverage run --source src/lightning_utilities -m pytest src/lightning_utilities tests -v
	python -m coverage report

	# specific file
	# python -m coverage run --source src/lightning_utilities -m pytest --flake8 --durations=0 -v -k

docs: clean
	pip install -e . -q -r requirements/_docs.txt
	cd docs && $(MAKE) html

clean:
	# clean all temp runs
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf build
	rm -rf dist
	rm -rf src/*.egg-info
	rm -rf ./docs/build
	rm -rf ./docs/source/**/generated
	rm -rf ./docs/source/api
