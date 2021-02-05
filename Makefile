.PHONY: test clean

test: clean
	# install APEX, see https://github.com/NVIDIA/apex#linux
	# to imitate SLURM set only single node
	export SLURM_LOCALID=0

	# use this to run tests
	python -m coverage run --source pytorch_lightning -m pytest pytorch_lightning tests pl_examples -v --flake8
	python -m coverage report -m

clean:
	# clean all temp runs
	rm -rf $(shell find . -name "mlruns")
	rm -rf $(shell find . -name "lightning_log")
	rm -rf _ckpt_*
