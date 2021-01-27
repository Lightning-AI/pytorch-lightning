.PHONY: test clean docs

test:
	# install APEX, see https://github.com/NVIDIA/apex#linux
	# to imitate SLURM set only single node
	export SLURM_LOCALID=0

	# use this to run tests
	rm -rf _ckpt_*
	rm -rf ./lightning_logs
	python -m coverage run --source pytorch_lightning -m pytest pytorch_lightning tests pl_examples -v --flake8
	python -m coverage report -m

	# specific file
	# python -m coverage run --source pytorch_lightning -m pytest --flake8 --durations=0 -v -k

docs:
	rm -rf ./docs/build
	rm -rf ./docs/source/generated
	rm -rf ./docs/source/*/generated
	rm -rf ./docs/source/api
	python -m sphinx -b html -D SPHINXOPTS="-W" docs/source docs/build

clean:
	# clean all temp runs
	rm -rf $(shell find . -name "mlruns")
