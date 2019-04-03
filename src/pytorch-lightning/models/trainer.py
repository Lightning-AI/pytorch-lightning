import torch
import tqdm
import numpy as np
from pytorch_lightning.root_module.memory import get_gpu_memory_map
import traceback
from pytorch_lightning.root_module.model_saving import TrainerIO
from torch.optim.lr_scheduler import MultiStepLR


class Trainer(TrainerIO):

    def __init__(self,
                 experiment,
                 checkpoint_callback, early_stop_callback,
                 cluster=None,
                 process_position=0,
                 current_gpu_name=0,
                 on_gpu=False,
                 enable_tqdm=True,
                 overfit_pct=0.0,
                 track_grad_norm=-1,
                 check_val_every_n_epoch=1,
                 fast_dev_run=False,
                 accumulate_grad_batches=1,
                 enable_early_stop=True, max_nb_epochs=5, min_nb_epochs=1,
                 train_percent_check=1.0, val_percent_check=1.0, test_percent_check=1.0, val_check_interval=0.95,
                 log_save_interval=1, add_log_row_interval=1,
                 lr_scheduler_milestones=None,
                 nb_sanity_val_steps=5):

        # Transfer params
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.enable_early_stop = enable_early_stop
        self.track_grad_norm = track_grad_norm
        self.fast_dev_run = fast_dev_run
        self.on_gpu = on_gpu
        self.enable_tqdm = enable_tqdm
        self.experiment = experiment
        self.exp_save_path = experiment.get_data_path(experiment.name, experiment.version)
        self.cluster = cluster
        self.process_position = process_position
        self.current_gpu_name = current_gpu_name
        self.checkpoint_callback = checkpoint_callback
        self.checkpoint_callback.save_function = self.save_checkpoint
        self.early_stop = early_stop_callback
        self.model = None
        self.max_nb_epochs = max_nb_epochs
        self.accumulate_grad_batches = accumulate_grad_batches
        self.early_stop_callback = early_stop_callback
        self.min_nb_epochs = min_nb_epochs
        self.nb_sanity_val_steps = nb_sanity_val_steps
        self.lr_scheduler_milestones = [] if lr_scheduler_milestones is None else [int(x.strip()) for x in lr_scheduler_milestones.split(',')]
        self.lr_schedulers = []

        # training state
        self.optimizers = None
        self.prog_bar = None
        self.global_step = 0
        self.current_epoch = 0
        self.total_batches = 0

        # logging
        self.log_save_interval = log_save_interval
        self.val_check_interval = val_check_interval
        self.add_log_row_interval = add_log_row_interval

        # dataloaders
        self.tng_dataloader = None
        self.test_dataloader = None
        self.val_dataloader = None

        # how much of the data to use
        self.__determine_data_use_amount(train_percent_check, val_percent_check, test_percent_check, overfit_pct)
        print('gpu available: {}, used: {}'.format(torch.cuda.is_available(), self.on_gpu))

    def __determine_data_use_amount(self, train_percent_check, val_percent_check, test_percent_check, overfit_pct):
        """
        Use less data for debugging purposes
        """
        self.train_percent_check = train_percent_check
        self.val_percent_check = val_percent_check
        self.test_percent_check = test_percent_check
        if overfit_pct > 0:
            self.train_percent_check = overfit_pct
            self.val_percent_check = overfit_pct
            self.test_percent_check = overfit_pct

    def __is_function_implemented(self, f_name):
        f_op = getattr(self, f_name, None)
        return callable(f_op)

    @property
    def __tng_tqdm_dic(self):
        tqdm_dic = {
            'tng_loss': '{0:.3f}'.format(self.avg_loss),
            'gpu': '{}'.format(self.current_gpu_name),
            'v_nb': '{}'.format(self.experiment.version),
            'epoch': '{}'.format(self.current_epoch),
            'batch_nb':'{}'.format(self.batch_nb),
        }
        tqdm_dic.update(self.tqdm_metrics)
        return tqdm_dic

    def __layout_bookeeping(self):
        # training bookeeping
        self.total_batch_nb = 0
        self.running_loss = []
        self.avg_loss = 0
        self.batch_nb = 0
        self.tqdm_metrics = {}

        # determine number of training batches
        nb_tng_batches = self.model.nb_batches(self.tng_dataloader)
        self.nb_tng_batches = int(nb_tng_batches * self.train_percent_check)

        # determine number of validation batches
        nb_val_batches = self.model.nb_batches(self.val_dataloader)
        nb_val_batches = int(nb_val_batches * self.val_percent_check)
        nb_val_batches = max(1, nb_val_batches)
        self.nb_val_batches = nb_val_batches

        # determine number of test batches
        nb_test_batches = self.model.nb_batches(self.test_dataloader)
        self.nb_test_batches = int(nb_test_batches * self.test_percent_check)

        # determine when to check validation
        self.val_check_batch = int(nb_tng_batches * self.val_check_interval)

    def __add_tqdm_metrics(self, metrics):
        for k, v in metrics.items():
            self.tqdm_metrics[k] = v

    def validate(self, model, dataloader, max_batches):
        """
        Run validation code
        :param model: PT model
        :param dataloader: PT dataloader
        :param max_batches: Scalar
        :return:
        """
        print('validating...')

        # enable eval mode
        model.zero_grad()
        model.eval()

        # disable gradients to save memory
        torch.set_grad_enabled(False)

        # bookkeeping
        outputs = []

        # run training
        for i, data_batch in enumerate(dataloader):

            if data_batch is None:
                continue

            # stop short when on fast dev run
            if max_batches is not None and i >= max_batches:
                break

            # -----------------
            # RUN VALIDATION STEP
            # -----------------
            output = model.validation_step(data_batch)
            outputs.append(output)

            # batch done
            if self.enable_tqdm and self.prog_bar is not None:
                self.prog_bar.update(1)

        # give model a chance to do something with the outputs
        val_results = model.validation_end(outputs)

        # enable train mode again
        model.train()

        # enable gradients to save memory
        torch.set_grad_enabled(True)
        return val_results

    def __get_dataloaders(self, model):
        """
        Dataloaders are provided by the model
        :param model:
        :return:
        """
        self.tng_dataloader = model.tng_dataloader
        self.test_dataloader = model.test_dataloader
        self.val_dataloader = model.val_dataloader

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    def fit(self, model):
        self.model = model

        # transfer data loaders from model
        self.__get_dataloaders(model)

        # init training constants
        self.__layout_bookeeping()

        # CHOOSE OPTIMIZER
        # filter out the weights that were done on gpu so we can load on good old cpus
        self.optimizers = model.configure_optimizers()

        # add lr schedulers
        if self.lr_scheduler_milestones is not None:
            for optimizer in self.optimizers:
                scheduler = MultiStepLR(optimizer, self.lr_scheduler_milestones)
                self.lr_schedulers.append(scheduler)

        # print model summary
        model.summarize()

        # put on gpu if needed
        if self.on_gpu:
            model = model.cuda()

        # run tiny validation to make sure program won't crash during val
        _ = self.validate(model, self.val_dataloader, max_batches=self.nb_sanity_val_steps)

        # save exp to get started
        self.experiment.save()

        # enable cluster checkpointing
        if self.cluster is not None:
            self.enable_auto_hpc_walltime_manager()

        # ---------------------------
        # CORE TRAINING LOOP
        # ---------------------------
        self.__train()

    def __train(self):
        # run all epochs
        for epoch_nb in range(self.current_epoch, self.max_nb_epochs):
            # update the lr scheduler
            for lr_scheduler in self.lr_schedulers:
                lr_scheduler.step()

            self.model.current_epoch = epoch_nb

            # hook
            if self.__is_function_implemented('on_epoch_start'):
                self.model.on_epoch_start()

            self.current_epoch = epoch_nb
            self.total_batches = self.nb_tng_batches + self.nb_val_batches
            self.batch_loss_value = 0  # accumulated grads

            # init progbar when requested
            if self.enable_tqdm:
                self.prog_bar = tqdm.tqdm(range(self.total_batches), position=self.process_position)

            for batch_nb, data_batch in enumerate(self.tng_dataloader):
                self.batch_nb = batch_nb
                self.global_step += 1
                self.model.global_step = self.global_step

                # stop when the flag is changed or we've gone past the amount requested in the batches
                self.total_batch_nb += 1
                met_batch_limit = batch_nb > self.nb_tng_batches
                if met_batch_limit:
                    break

                # ---------------
                # RUN TRAIN STEP
                # ---------------
                self.__run_tng_batch(data_batch)

                # ---------------
                # RUN VAL STEP
                # ---------------
                is_val_check_batch = (batch_nb + 1) % self.val_check_batch == 0
                if self.fast_dev_run or is_val_check_batch:
                    self.__run_validation()

                # when batch should be saved
                if (batch_nb + 1) % self.log_save_interval == 0:
                    self.experiment.save()

                # when metrics should be logged
                if batch_nb % self.add_log_row_interval == 0:
                    # count items in memory
                    # nb_params, nb_tensors = count_mem_items()

                    metrics = self.model.update_tng_log_metrics(self.__tng_tqdm_dic)

                    # add gpu memory
                    if self.on_gpu:
                        mem_map = get_gpu_memory_map()
                        metrics.update(mem_map)

                    # add norms
                    if self.track_grad_norm > 0:
                        grad_norm_dic = self.model.grad_norm(self.track_grad_norm)
                        metrics.update(grad_norm_dic)

                    # log metrics
                    self.experiment.log(metrics)
                    self.experiment.save()

                # hook
                if self.__is_function_implemented('on_batch_end'):
                    self.model.on_batch_end()

            # hook
            if self.__is_function_implemented('on_epoch_end'):
                self.model.on_epoch_end()

            # early stopping
            if self.enable_early_stop:
                should_stop = self.early_stop_callback.on_epoch_end(epoch=epoch_nb, logs=self.__tng_tqdm_dic)
                met_min_epochs = epoch_nb > self.min_nb_epochs

                # stop training
                stop = should_stop and met_min_epochs
                if stop:
                    return

    def __run_tng_batch(self, data_batch):
        if data_batch is None:
            return

        # hook
        if self.__is_function_implemented('on_batch_start'):
            self.model.on_batch_start()

        if self.enable_tqdm:
            self.prog_bar.update(1)

        # forward pass
        # return a scalar value and a dic with tqdm metrics
        loss, model_specific_tqdm_metrics_dic = self.model.training_step(data_batch)
        self.__add_tqdm_metrics(model_specific_tqdm_metrics_dic)

        # backward pass
        loss.backward()
        self.batch_loss_value += loss.item()

        # gradient update with accumulated gradients
        if (self.batch_nb + 1) % self.accumulate_grad_batches == 0:

            # update gradients across all optimizers
            for optimizer in self.optimizers:
                optimizer.step()

                # clear gradients
                optimizer.zero_grad()

            # queuing loss across batches blows it up proportionally... divide out the number accumulated
            self.batch_loss_value = self.batch_loss_value / self.accumulate_grad_batches

            # track loss
            self.running_loss.append(self.batch_loss_value)
            self.batch_loss_value = 0
            self.avg_loss = np.mean(self.running_loss[-100:])

            # update progbar
            if self.enable_tqdm:
                # add model specific metrics
                tqdm_metrics = self.__tng_tqdm_dic
                self.prog_bar.set_postfix(**tqdm_metrics)

        # activate batch end hook
        if self.__is_function_implemented('on_batch_end'):
            self.model.on_batch_end()

    def __run_validation(self):
        # decide if can check epochs
        can_check_epoch = (self.current_epoch + 1) % self.check_val_every_n_epoch == 0
        if self.fast_dev_run:
            print('skipping to check performance bc of --fast_dev_run')
        elif not can_check_epoch:
            return

        try:
            # hook
            if self.__is_function_implemented('on_pre_performance_check'):
                self.model.on_pre_performance_check()

            # use full val set on end of epoch
            # use a small portion otherwise
            max_batches = None if not self.fast_dev_run else 1
            model_specific_tqdm_metrics_dic = self.validate(
                self.model,
                self.val_dataloader,
                max_batches
            )
            self.__add_tqdm_metrics(model_specific_tqdm_metrics_dic)

            # hook
            if self.__is_function_implemented('on_post_performance_check'):
                self.model.on_post_performance_check()

        except Exception as e:
            print(e)
            print(traceback.print_exc())

        if self.enable_tqdm:
            # add model specific metrics
            tqdm_metrics = self.__tng_tqdm_dic
            self.prog_bar.set_postfix(**tqdm_metrics)

        # model checkpointing
        print('save callback...')
        self.checkpoint_callback.on_epoch_end(epoch=self.current_epoch, logs=self.__tng_tqdm_dic)
