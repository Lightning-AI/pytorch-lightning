import torch
import tqdm
import numpy as np
from pytorch_lightning.root_module.memory import get_gpu_memory_map
import traceback
from pytorch_lightning.root_module.model_saving import TrainerIO
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_lightning.pt_overrides.override_data_parallel import LightningDataParallel
import pdb

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


def reduce_distributed_output(output, nb_gpus):
    for k, v in output.items():
        # recurse on nested dics
        if isinstance(output[k], dict):
            output[k] = reduce_distributed_output(output[k], nb_gpus)

        # reduce only metrics that have the same nb of gpus
        elif output[k].size(0) == nb_gpus:
            reduced = torch.mean(output[k])
            output[k] = reduced
    return output


class Trainer(TrainerIO):

    def __init__(self,
                 experiment,
                 checkpoint_callback, early_stop_callback,
                 gradient_clip=0,
                 cluster=None,
                 process_position=0,
                 current_gpu_name=0,
                 gpus=None,
                 progress_bar=True,
                 overfit_pct=0.0,
                 track_grad_norm=-1,
                 check_val_every_n_epoch=1,
                 fast_dev_run=False,
                 accumulate_grad_batches=1,
                 enable_early_stop=True, max_nb_epochs=1000, min_nb_epochs=1,
                 train_percent_check=1.0, val_percent_check=1.0, test_percent_check=1.0, val_check_interval=0.95,
                 log_save_interval=100, add_log_row_interval=10,
                 lr_scheduler_milestones=None,
                 use_amp=False,
                 print_nan_grads=False,
                 amp_level='O2',
                 nb_sanity_val_steps=5):

        # Transfer params
        self.gradient_clip = gradient_clip
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.enable_early_stop = enable_early_stop
        self.track_grad_norm = track_grad_norm
        self.fast_dev_run = fast_dev_run
        self.on_gpu = gpus is not None and torch.cuda.is_available()
        self.progress_bar = progress_bar
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
        self.amp_level = amp_level
        self.print_nan_grads = print_nan_grads
        self.data_parallel_device_ids = gpus
        self.data_parallel = gpus is not None and len(gpus) > 0

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

        # apex test
        self.use_amp = use_amp and APEX_AVAILABLE
        if self.use_amp:
            print('using 16bit precision')

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
        f_op = getattr(self.model, f_name, None)
        return callable(f_op)

    @property
    def __tng_tqdm_dic(self):
        tqdm_dic = {
            'tng_loss': '{0:.3f}'.format(self.avg_loss),
            'v_nb': '{}'.format(self.experiment.version),
            'epoch': '{}'.format(self.current_epoch),
            'batch_nb':'{}'.format(self.batch_nb),
        }
        tqdm_dic.update(self.tqdm_metrics)

        if self.on_gpu:
            tqdm_dic['gpu'] = '{}'.format(self.current_gpu_name)

        return tqdm_dic

    def __layout_bookeeping(self, model):
        # training bookeeping
        self.total_batch_nb = 0
        self.running_loss = []
        self.avg_loss = 0
        self.batch_nb = 0
        self.tqdm_metrics = {}

        # determine number of training batches
        self.nb_tng_batches = model.nb_batches(self.tng_dataloader)
        self.nb_tng_batches = int(self.nb_tng_batches * self.train_percent_check)

        # determine number of validation batches
        self.nb_val_batches = model.nb_batches(self.val_dataloader)
        self.nb_val_batches = int(self.nb_val_batches * self.val_percent_check)
        self.nb_val_batches = max(1, self.nb_val_batches)
        self.nb_val_batches = self.nb_val_batches

        # determine number of test batches
        self.nb_test_batches = model.nb_batches(self.test_dataloader)
        self.nb_test_batches = int(self.nb_test_batches * self.test_percent_check)

        # determine when to check validation
        self.val_check_batch = int(self.nb_tng_batches * self.val_check_interval)

    def __add_tqdm_metrics(self, metrics):
        for k, v in metrics.items():
            if type(v) is torch.Tensor:
                v = v.item()

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
        model.from_lightning = True

        # disable gradients to save memory
        torch.set_grad_enabled(False)

        # bookkeeping
        outputs = []

        # run training
        for batch_i, data_batch in enumerate(dataloader):

            if data_batch is None:
                continue

            # stop short when on fast dev run
            if max_batches is not None and batch_i >= max_batches:
                break

            # -----------------
            # RUN VALIDATION STEP
            # -----------------
            if self.data_parallel:
                output = model(data_batch, batch_i)
                output = reduce_distributed_output(output, len(self.data_parallel_device_ids))
            else:
                output = model.validation_step(data_batch, batch_i)

            outputs.append(output)

            # batch done
            if self.progress_bar and self.prog_bar is not None:
                self.prog_bar.update(1)

        # give model a chance to do something with the outputs
        if self.data_parallel:
            val_results = model.module.validation_end(outputs)
        else:
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

        # give model convenience properties
        model.trainer = self
        model.experiment = self.experiment

        # transfer data loaders from model
        self.__get_dataloaders(model)

        # init training constants
        self.__layout_bookeeping(model)

        # CHOOSE OPTIMIZER
        # filter out the weights that were done on gpu so we can load on good old cpus
        self.optimizers = model.configure_optimizers()

        if self.use_amp:
            # An example
            model, optimizer = amp.initialize(
                model, self.optimizers[0], opt_level=self.amp_level,
            )
            self.optimizers[0] = optimizer
            model.trainer = self

        # add lr schedulers
        if self.lr_scheduler_milestones is not None:
            for optimizer in self.optimizers:
                scheduler = MultiStepLR(optimizer, self.lr_scheduler_milestones)
                self.lr_schedulers.append(scheduler)

        # print model summary
        model.summarize()

        # put on gpu if needed
        if self.on_gpu:
            model = LightningDataParallel(model, device_ids=self.data_parallel_device_ids)
            model.cuda(self.data_parallel_device_ids[0])

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
        self.model = model
        self.__train()

    def __train(self):
        # run all epochs
        for epoch_nb in range(self.current_epoch, self.max_nb_epochs):
            # update the lr scheduler
            for lr_scheduler in self.lr_schedulers:
                lr_scheduler.step()

            model = self.model.module if self.data_parallel else self.model
            model.current_epoch = epoch_nb

            # hook
            if self.__is_function_implemented('on_epoch_start'):
                model = self.model.module if self.data_parallel else self.model
                model.on_epoch_start()

            self.current_epoch = epoch_nb
            self.total_batches = self.nb_tng_batches + self.nb_val_batches
            self.batch_loss_value = 0  # accumulated grads

            # init progbar when requested
            if self.progress_bar:
                self.prog_bar = tqdm.tqdm(range(self.total_batches), position=self.process_position)

            for batch_nb, data_batch in enumerate(self.tng_dataloader):
                self.batch_nb = batch_nb
                self.global_step += 1

                model = self.model.module if self.data_parallel else self.model
                model.global_step = self.global_step

                # stop when the flag is changed or we've gone past the amount requested in the batches
                self.total_batch_nb += 1
                met_batch_limit = batch_nb > self.nb_tng_batches
                if met_batch_limit:
                    break

                # ---------------
                # RUN TRAIN STEP
                # ---------------
                batch_result = self.__run_tng_batch(data_batch, batch_nb)
                early_stop_epoch = batch_result == -1

                # ---------------
                # RUN VAL STEP
                # ---------------
                is_val_check_batch = (batch_nb + 1) % self.val_check_batch == 0
                if self.fast_dev_run or is_val_check_batch or early_stop_epoch:
                    self.__run_validation()

                # when batch should be saved
                if (batch_nb + 1) % self.log_save_interval == 0 or early_stop_epoch:
                    self.experiment.save()

                # when metrics should be logged
                if batch_nb % self.add_log_row_interval == 0 or early_stop_epoch:
                    # count items in memory
                    # nb_params, nb_tensors = count_mem_items()

                    if self.data_parallel:
                        metrics = self.model.module.update_tng_log_metrics(self.__tng_tqdm_dic)
                    else:
                        metrics = self.model.update_tng_log_metrics(self.__tng_tqdm_dic)

                    # add gpu memory
                    if self.on_gpu:
                        mem_map = get_gpu_memory_map()
                        metrics.update(mem_map)

                    # add norms
                    if self.track_grad_norm > 0:
                        model = self.model.module if self.data_parallel else self.model
                        grad_norm_dic = model.grad_norm(self.track_grad_norm)

                        metrics.update(grad_norm_dic)

                    # log metrics
                    scalar_metrics = self.__metrics_to_scalars(metrics, blacklist=self.__log_vals_blacklist())
                    self.experiment.log(scalar_metrics, global_step=self.global_step)
                    self.experiment.save()

                # hook
                if self.__is_function_implemented('on_batch_end'):
                    model = self.model.module if self.data_parallel else self.model
                    model.on_batch_end()

                # end epoch early
                if early_stop_epoch:
                    break

            # hook
            if self.__is_function_implemented('on_epoch_end'):
                model = self.model.module if self.data_parallel else self.model
                model.on_epoch_end()

            # early stopping
            if self.enable_early_stop:
                should_stop = self.early_stop_callback.on_epoch_end(epoch=epoch_nb, logs=self.__tng_tqdm_dic)
                met_min_epochs = epoch_nb > self.min_nb_epochs

                # stop training
                stop = should_stop and met_min_epochs
                if stop:
                    return

    def __metrics_to_scalars(self, metrics, blacklist=[]):
        new_metrics = {}
        for k, v in metrics.items():
            if type(v) is torch.Tensor:
                v = v.item()

            if type(v) is dict:
                v = self.__metrics_to_scalars(v)

            if k not in blacklist:
                new_metrics[k] = float(v)

        return new_metrics

    def __log_vals_blacklist(self):
        """avoid logging some vals lightning uses to maintain state"""
        blacklist = {'batch_nb', 'v_nb', 'epoch', 'gpu'}
        return blacklist

    def __run_tng_batch(self, data_batch, batch_nb):
        if data_batch is None:
            return 0

        # hook
        if self.__is_function_implemented('on_batch_start'):
            model = self.model.module if self.data_parallel else self.model
            response = model.on_batch_start(data_batch)

            if response == -1:
                return -1

        if self.progress_bar:
            self.prog_bar.update(1)

        # forward pass
        # return a scalar value and a dic with tqdm metrics
        if self.data_parallel:
            output = self.model(data_batch, batch_nb)
            output = reduce_distributed_output(output, len(self.data_parallel_device_ids))
        else:
            output = self.model.training_step(data_batch, batch_nb)

        model_specific_tqdm_metrics_dic = output['tqdm_metrics']
        loss = output['loss']

        self.__add_tqdm_metrics(model_specific_tqdm_metrics_dic)

        # backward pass
        if self.use_amp:
            for optimizer in self.optimizers:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
        else:
            loss.backward()

        if self.print_nan_grads:
            model = self.model.module if self.data_parallel else self.model
            for param in model.parameters():
                print(param.grad.float().sum())

        self.batch_loss_value += loss.item()

        # gradient update with accumulated gradients
        if (self.batch_nb + 1) % self.accumulate_grad_batches == 0:

            # clip gradients
            if self.gradient_clip > 0:
                model = self.model.module if self.data_parallel else self.model
                torch.nn.utils.clip_grad_norm(model.parameters(), self.gradient_clip)

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
            if self.progress_bar:
                # add model specific metrics
                tqdm_metrics = self.__tng_tqdm_dic
                self.prog_bar.set_postfix(**tqdm_metrics)

        # activate batch end hook
        if self.__is_function_implemented('on_batch_end'):
            self.model.on_batch_end()

        return 0

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

        if self.progress_bar:
            # add model specific metrics
            tqdm_metrics = self.__tng_tqdm_dic
            self.prog_bar.set_postfix(**tqdm_metrics)

        # model checkpointing
        print('save callback...')
        self.checkpoint_callback.on_epoch_end(epoch=self.current_epoch, logs=self.__tng_tqdm_dic)
