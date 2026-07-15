.PHONY: test clean docs setup standalone

# to imitate SLURM set only single node
export SLURM_LOCALID=0
# assume you have installed need packages
export SPHINX_MOCK_REQUIREMENTS=1
# install only Lightning Trainer packages
export PACKAGE_NAME=pytorch

STANDALONE_DIR:=standalone_artifacts
STANDALONE_SCRIPT:=$(STANDALONE_DIR)/run_standalone_tests.sh


# In Lightning Studio, the `lightning` package comes pre-installed.
# Uninstall it first to ensure the editable install works correctly.
setup: update
	uv pip uninstall lightning pytorch-lightning lightning-fabric || true
	uv pip install -r requirements.txt \
	    -r requirements/pytorch/base.txt \
	    -r requirements/pytorch/test.txt \
		-r requirements/pytorch/test_gpu.txt \
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


# Run standalone tests
#
# Usage:
#   make standalone TEST=tests/tests_pytorch
#   make standalone TEST=tests/tests_pytorch/test_file::test_name
#
standalone: _download_standalone_script
	@if [ -z "$(TEST)" ]; then \
		echo "\033[0;31mERROR: TEST variable is not set. Please provide a test to run.\033[0m"; \
		exit 1; \
	else \
		echo "----- running tests: '$(TEST)' -----"; \
		export PL_RUN_STANDALONE_TESTS=1; \
		export STANDALONE_ARTIFACTS_DIR=$(STANDALONE_DIR); \
		./$(STANDALONE_SCRIPT) $(TEST); \
		status=$$?; \
		if [ $$status -ne 0 ]; then \
			echo "\033[0;31m----- Standalone tests failed :( -----\033[0m"; \
			exit 0; \
		else \
			echo "\033[0;32m----- Standalone tests passed! ;) -----\033[0m"; \
			exit 0; \
		fi; \
	fi

_download_standalone_script:
	@mkdir -p $(STANDALONE_DIR); \
	if [ ! -f $(STANDALONE_SCRIPT) ]; then \
		echo "Downloading standalone test script..."; \
		curl -fsSL https://raw.githubusercontent.com/Lightning-AI/utilities/main/scripts/run_standalone_tests.sh -o $(STANDALONE_SCRIPT); \
		chmod +x $(STANDALONE_SCRIPT); \
		echo "Standalone test script downloaded to $(STANDALONE_SCRIPT)"; \
	else \
		echo "----- Standalone script already exists. -----"; \
	fi
