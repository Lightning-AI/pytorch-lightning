.PHONY: test clean docs setup

# to imitate SLURM set only single node
export SLURM_LOCALID=0
# assume you have installed need packages
export SPHINX_MOCK_REQUIREMENTS=1
# install only Lightning Trainer packages
export PACKAGE_NAME=pytorch


# In Lightning Studio, the `lightning` package comes pre-installed.
# Uninstall it first to ensure the editable install works correctly.
setup: update
	uv pip uninstall lightning pytorch-lightning lightning-fabric || true
	uv pip install -r requirements.txt \
	    -r requirements/pytorch/base.txt \
	    -r requirements/pytorch/test.txt \
	    -r requirements/pytorch/extra.txt \
	    -r requirements/pytorch/strategies.txt \
	    -r requirements/fabric/base.txt \
	    -r requirements/fabric/test.txt \
	    -r requirements/fabric/strategies.txt \
	    -r requirements/typing.txt \
	    -e ".[all]" \
	    pre-commit
	pre-commit install
	@echo "-----------------------------"
	@echo "✅ Environment setup complete. Ready to Contribute ⚡️!"


clean:
	# clean all temp runs
	rm -rf $(shell find . -name "mlruns")
	rm -rf $(shell find . -name "lightning_log")
	rm -rf $(shell find . -name "lightning_logs")
	rm -rf _ckpt_*
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf ./docs/build
	rm -rf ./docs/source-fabric/api/generated
	rm -rf ./docs/source-pytorch/notebooks
	rm -rf ./docs/source-pytorch/generated
	rm -rf ./docs/source-pytorch/*/generated
	rm -rf ./docs/source-pytorch/api
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	rm -rf src/lightning_fabric/*/
	rm -rf src/pytorch_lightning/*/

test: clean setup
	# Review the CONTRIBUTING documentation for other ways to test.

	# run tests with coverage
	python -m coverage run --source src/lightning/pytorch -m pytest src/lightning/pytorch tests/tests_pytorch -v
	python -m coverage run --source src/lightning/fabric -m pytest src/lightning/fabric tests/tests_fabric -v
	python -m coverage report

docs: docs-pytorch

sphinx-theme: setup
	uv pip install -q awscli
	mkdir -p dist/
	aws s3 sync --no-sign-request s3://sphinx-packages/ dist/
	uv pip install lai-sphinx-theme -f dist/

docs-fabric: clean sphinx-theme
	uv pip install -e '.[all]' --quiet -r requirements/fabric/docs.txt
	cd docs/source-fabric && $(MAKE) html --jobs $(nproc)

docs-pytorch: clean sphinx-theme
	uv pip install -e '.[all]' --quiet -r requirements/pytorch/docs.txt
	cd docs/source-pytorch && $(MAKE) html --jobs $(nproc)

update:
	git submodule update --init --recursive --remote
