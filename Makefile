.PHONY: test clean

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
	# python -m coverage run --source pytorch_lightning -m py.test --flake8 --durations=0 -v -k

clean:
	# clean all temp runs
	rm -rf $(shell find . -name "mlruns" )
