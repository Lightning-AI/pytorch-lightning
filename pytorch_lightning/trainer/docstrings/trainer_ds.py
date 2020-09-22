init = r"""

        Customize every aspect of training via flags

        Args:
            logger: Logger (or iterable collection of loggers) for experiment tracking.

            checkpoint_callback: Callback for checkpointing.

            early_stop_callback (:class:`pytorch_lightning.callbacks.EarlyStopping`):

            callbacks: Add a list of callbacks.

            default_root_dir: Default path for logs and weights when no logger/ckpt_callback passed.
                Default: ``os.getcwd()``.
                Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'

            gradient_clip_val: 0 means don't clip.

            process_position: orders the progress bar when running multiple models on same machine.

            num_nodes: number of GPU nodes for distributed training.

            gpus: number of gpus to train on (int) or which GPUs to train on (list or str) applied per node

            auto_select_gpus:

                If enabled and `gpus` is an integer, pick available
                gpus automatically. This is especially useful when
                GPUs are configured to be in "exclusive mode", such
                that only one process at a time can access them.

            tpu_cores: How many TPU cores to train on (1 or 8) / Single TPU to train on [1]

            log_gpu_memory: None, 'min_max', 'all'. Might slow performance

            progress_bar_refresh_rate: How often to refresh progress bar (in steps). Value ``0`` disables progress bar.
                Ignored when a custom callback is passed to :paramref:`~Trainer.callbacks`.

            overfit_batches: Overfit a percent of training data (float) or a set number of batches (int). Default: 0.0

            overfit_pct:
                .. warning:: .. deprecated:: 0.8.0

                    Use `overfit_batches` instead. Will be removed in 0.10.0.

            track_grad_norm: -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf' infinity-norm.

            check_val_every_n_epoch: Check val every n train epochs.

            fast_dev_run: runs 1 batch of train, test and val to find any bugs (ie: a sort of unit test).

            accumulate_grad_batches: Accumulates grads every k batches or as set up in the dict.

            max_epochs: Stop training once this number of epochs is reached.

            min_epochs: Force training for at least these many epochs

            max_steps: Stop training after this number of steps. Disabled by default (None).

            min_steps: Force training for at least these number of steps. Disabled by default (None).

            limit_train_batches: How much of training dataset to check (floats = percent, int = num_batches)

            limit_val_batches: How much of validation dataset to check (floats = percent, int = num_batches)

            limit_test_batches: How much of test dataset to check (floats = percent, int = num_batches)

            val_check_interval: How often to check the validation set. Use float to check within a training epoch,
                use int to check every n steps (batches).

            log_save_interval: Writes logs to disk this often

            row_log_interval: How often to add logging rows (does not write to disk)

            distributed_backend: The distributed backend to use (dp, ddp, ddp2, ddp_spawn, ddp_cpu)

            sync_batchnorm: Synchronize batch norm layers between process groups/whole world.

            precision: Full precision (32), half precision (16). Can be used on CPU, GPU or TPUs.

            weights_summary: Prints a summary of the weights when training begins.

            weights_save_path: Where to save weights if specified. Will override default_root_dir
                    for checkpoints only. Use this if for whatever reason you need the checkpoints
                    stored in a different place than the logs written in `default_root_dir`.
                    Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
                    Defaults to `default_root_dir`.

            amp_backend: The mixed precision backend to use ("native" or "apex")

            amp_level: The optimization level to use (O1, O2, etc...).

            num_sanity_val_steps: Sanity check runs n validation batches before starting the training routine.
                Set it to `-1` to run all batches in all validation dataloaders. Default: 2

            truncated_bptt_steps: Truncated back prop breaks performs backprop every k steps of much longer
                sequence.

            resume_from_checkpoint: To resume training from a specific checkpoint pass in the path here.
                This can be a URL.

            profiler:  To profile individual steps during training and assist in identifying bottlenecks.

            reload_dataloaders_every_epoch: Set to True to reload dataloaders every epoch.

            auto_lr_find: If set to True, will `initially` run a learning rate finder,
                trying to optimize initial learning for faster convergence. Sets learning
                rate in self.lr or self.learning_rate in the LightningModule.
                To use a different key, set a string instead of True with the key name.

            replace_sampler_ddp: Explicitly enables or disables sampler replacement. If not specified this
                will toggled automatically when DDP is used. By default it will add ``shuffle=True`` for
                train sampler and ``shuffle=False`` for val/test sampler. If you want to customize it,
                you can set ``replace_sampler_ddp=False`` and add your own distributed sampler.

            benchmark: If true enables cudnn.benchmark.

            deterministic: If true enables cudnn.deterministic.

            terminate_on_nan: If set to True, will terminate training (by raising a `ValueError`) at the
                end of each training batch, if any of the parameters or the loss are NaN or +/-inf.

            auto_scale_batch_size: If set to True, will `initially` run a batch size
                finder trying to find the largest batch size that fits into memory.
                The result will be stored in self.batch_size in the LightningModule.
                Additionally, can be set to either `power` that estimates the batch size through
                a power search or `binsearch` that estimates the batch size through a binary search.

            prepare_data_per_node: If True, each LOCAL_RANK=0 will call prepare data.
                Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data
        """

fit = r"""
        Runs the full optimization routine.

        Args:
            model: Model to fit.

            train_dataloader: A Pytorch
                DataLoader with training samples. If the model has
                a predefined train_dataloader method this will be skipped.

            val_dataloaders: Either a single
                Pytorch Dataloader or a list of them, specifying validation samples.
                If the model has a predefined val_dataloaders method this will be skipped

        Example::

            # Option 1,
            # Define the train_dataloader() and val_dataloader() fxs
            # in the lightningModule
            # RECOMMENDED FOR MOST RESEARCH AND APPLICATIONS TO MAINTAIN READABILITY
            trainer = Trainer()
            model = LightningModule()
            trainer.fit(model)

            # Option 2
            # in production cases we might want to pass different datasets to the same model
            # Recommended for PRODUCTION SYSTEMS
            train, val = DataLoader(...), DataLoader(...)
            trainer = Trainer()
            model = LightningModule()
            trainer.fit(model, train_dataloader=train, val_dataloaders=val)

            # Option 1 & 2 can be mixed, for example the training set can be
            # defined as part of the model, and validation can then be feed to .fit()

        """

test = r"""

        Separates from fit to make sure you never run on your test set until you want to.

        Args:
            model: The model to test.

            test_dataloaders: Either a single
                Pytorch Dataloader or a list of them, specifying validation samples.

            ckpt_path: Either ``best`` or path to the checkpoint you wish to test.
                If ``None``, use the weights from the last epoch to test. Default to ``best``.

            verbose: If True, prints the test results

        Returns:
            The final test result dictionary. If no test_epoch_end is defined returns a list of dictionaries

        Example::

            # Option 1
            # run test with the best checkpoint from ``ModelCheckpoint`` after fitting.
            test = DataLoader(...)
            trainer = Trainer()
            model = LightningModule()

            trainer.fit(model)
            trainer.test(test_dataloaders=test)

            # Option 2
            # run test with the specified checkpoint after fitting
            test = DataLoader(...)
            trainer = Trainer()
            model = LightningModule()

            trainer.fit(model)
            trainer.test(test_dataloaders=test, ckpt_path='path/to/checkpoint.ckpt')

            # Option 3
            # run test with the weights from the end of training after fitting
            test = DataLoader(...)
            trainer = Trainer()
            model = LightningModule()

            trainer.fit(model)
            trainer.test(test_dataloaders=test, ckpt_path=None)

            # Option 4
            # run test from a loaded model. ``ckpt_path`` is ignored in this case.
            test = DataLoader(...)
            model = LightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')
            trainer = Trainer()
            trainer.test(model, test_dataloaders=test)
        """
