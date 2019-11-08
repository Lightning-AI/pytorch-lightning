import numpy as np
import tqdm

try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


class TrainerTrainLoopMixin(object):

    def train(self):
        # run all epochs
        for epoch_nb in range(self.current_epoch, self.max_nb_epochs):
            # set seed for distributed sampler (enables shuffling for each epoch)
            if self.use_ddp and hasattr(self.get_train_dataloader().sampler, 'set_epoch'):
                self.get_train_dataloader().sampler.set_epoch(epoch_nb)

            # get model
            model = self.get_model()

            # update training progress in trainer and model
            model.current_epoch = epoch_nb
            self.current_epoch = epoch_nb

            # val can be checked multiple times in epoch
            is_val_epoch = (self.current_epoch + 1) % self.check_val_every_n_epoch == 0
            val_checks_per_epoch = self.nb_training_batches // self.val_check_batch
            val_checks_per_epoch = val_checks_per_epoch if is_val_epoch else 0

            # total batches includes multiple val checks
            self.total_batches = (self.nb_training_batches +
                                  self.nb_val_batches * val_checks_per_epoch)
            self.batch_loss_value = 0  # accumulated grads

            if self.fast_dev_run:
                # limit the number of batches to 2 (1 train and 1 val) in fast_dev_run
                nb_iterations = 2
            elif self.is_iterable_train_dataloader:
                # for iterable train loader, the progress bar never ends
                nb_iterations = None
            else:
                nb_iterations = self.total_batches

            # reset progress bar
            # .reset() doesn't work on disabled progress bar so we should check
            if not self.main_progress_bar.disable:
                self.main_progress_bar.reset(nb_iterations)
            desc = f'Epoch {epoch_nb + 1}' if not self.is_iterable_train_dataloader else ''
            self.main_progress_bar.set_description(desc)

            # changing gradient according accumulation_scheduler
            self.accumulation_scheduler.on_epoch_begin(epoch_nb, self)

            # -----------------
            # RUN TNG EPOCH
            # -----------------
            self.run_training_epoch()

            # update LR schedulers
            if self.lr_schedulers is not None:
                for lr_scheduler in self.lr_schedulers:
                    lr_scheduler.step(self.current_epoch)

            # early stopping
            met_min_epochs = epoch_nb > self.min_nb_epochs
            if self.enable_early_stop and (met_min_epochs or self.fast_dev_run):
                should_stop = self.early_stop_callback.on_epoch_end(epoch=epoch_nb,
                                                                    logs=self.callback_metrics)
                # stop training
                stop = should_stop and met_min_epochs
                if stop:
                    self.main_progress_bar.close()
                    return

        self.main_progress_bar.close()

        if self.logger is not None:
            self.logger.finalize("success")

    def run_training_epoch(self):
        # before epoch hook
        if self.is_function_implemented('on_epoch_start'):
            model = self.get_model()
            model.on_epoch_start()

        # run epoch
        for batch_nb, batch in enumerate(self.get_train_dataloader()):
            self.batch_nb = batch_nb

            model = self.get_model()
            model.global_step = self.global_step

            # ---------------
            # RUN TRAIN STEP
            # ---------------
            output = self.run_training_batch(batch, batch_nb)
            batch_result, grad_norm_dic, batch_step_metrics = output

            # when returning -1 from train_step, we end epoch early
            early_stop_epoch = batch_result == -1

            # ---------------
            # RUN VAL STEP
            # ---------------
            is_val_check_batch = (batch_nb + 1) % self.val_check_batch == 0
            can_check_epoch = (self.current_epoch + 1) % self.check_val_every_n_epoch == 0
            should_check_val = ((is_val_check_batch or early_stop_epoch) and can_check_epoch)

            # fast_dev_run always forces val checking after train batch
            if self.fast_dev_run or should_check_val:
                self.run_evaluation(test=self.testing)

            # when logs should be saved
            should_save_log = (batch_nb + 1) % self.log_save_interval == 0 or early_stop_epoch
            if should_save_log or self.fast_dev_run:
                if self.proc_rank == 0 and self.logger is not None:
                    self.logger.save()

            # when metrics should be logged
            should_log_metrics = batch_nb % self.row_log_interval == 0 or early_stop_epoch
            if should_log_metrics or self.fast_dev_run:
                # logs user requested information to logger
                self.log_metrics(batch_step_metrics, grad_norm_dic)

            self.global_step += 1
            self.total_batch_nb += 1

            # end epoch early
            # stop when the flag is changed or we've gone past the amount
            # requested in the batches
            if early_stop_epoch or self.fast_dev_run:
                break

            # stop epoch if we limited nb batches
            met_batch_limit = batch_nb >= self.nb_training_batches
            if met_batch_limit:
                break

        # epoch end hook
        if self.is_function_implemented('on_epoch_end'):
            model = self.get_model()
            model.on_epoch_end()

    def run_training_batch(self, batch, batch_nb):
        # track grad norms
        grad_norm_dic = {}

        # track all metrics for callbacks
        all_callback_metrics = []

        # track metrics to log
        all_log_metrics = []

        if batch is None:
            return 0, grad_norm_dic

        # hook
        if self.is_function_implemented('on_batch_start'):
            model_ref = self.get_model()
            response = model_ref.on_batch_start(batch)

            if response == -1:
                return -1, grad_norm_dic

        splits = [batch]
        if self.truncated_bptt_steps is not None:
            model_ref = self.get_model()
            splits = model_ref.tbptt_split_batch(batch, self.truncated_bptt_steps)

        self.hiddens = None
        for split_nb, split_batch in enumerate(splits):
            self.split_nb = split_nb

            # call training_step once per optimizer
            for opt_idx, optimizer in enumerate(self.optimizers):

                # wrap the forward step in a closure so second order methods work
                def optimizer_closure():
                    # forward pass
                    output = self.training_forward(
                        split_batch, batch_nb, opt_idx, self.hiddens)

                    closure_loss = output[0]
                    progress_bar_metrics = output[1]
                    log_metrics = output[2]
                    callback_metrics = output[3]
                    self.hiddens = output[4]

                    # accumulate loss
                    # (if accumulate_grad_batches = 1 no effect)
                    closure_loss = closure_loss / self.accumulate_grad_batches

                    # backward pass
                    model_ref = self.get_model()
                    model_ref.backward(self.use_amp, closure_loss, optimizer)

                    # track metrics for callbacks
                    all_callback_metrics.append(callback_metrics)

                    # track progress bar metrics
                    self.add_tqdm_metrics(progress_bar_metrics)
                    all_log_metrics.append(log_metrics)

                    # insert after step hook
                    if self.is_function_implemented('on_after_backward'):
                        model_ref = self.get_model()
                        model_ref.on_after_backward()

                    return closure_loss

                # calculate loss
                loss = optimizer_closure()

                # nan grads
                if self.print_nan_grads:
                    self.print_nan_gradients()

                # track total loss for logging (avoid mem leaks)
                self.batch_loss_value += loss.item()

                # gradient update with accumulated gradients
                if (self.batch_nb + 1) % self.accumulate_grad_batches == 0:

                    # track gradient norms when requested
                    if batch_nb % self.row_log_interval == 0:
                        if self.track_grad_norm > 0:
                            model = self.get_model()
                            grad_norm_dic = model.grad_norm(
                                self.track_grad_norm)

                    # clip gradients
                    self.clip_gradients()

                    # calls .step(), .zero_grad()
                    # override function to modify this behavior
                    model = self.get_model()
                    model.optimizer_step(self.current_epoch, batch_nb,
                                         optimizer, opt_idx, optimizer_closure)

                    # calculate running loss for display
                    self.running_loss.append(self.batch_loss_value)
                    self.batch_loss_value = 0
                    self.avg_loss = np.mean(self.running_loss[-100:])

        # activate batch end hook
        if self.is_function_implemented('on_batch_end'):
            model = self.get_model()
            model.on_batch_end()

        # update progress bar
        self.main_progress_bar.update(1)
        self.main_progress_bar.set_postfix(**self.training_tqdm_dict)

        # collapse all metrics into one dict
        all_log_metrics = {k: v for d in all_log_metrics for k, v in d.items()}

        # track all metrics for callbacks
        self.callback_metrics = {k: v for d in all_callback_metrics for k, v in d.items()}

        return 0, grad_norm_dic, all_log_metrics

    def training_forward(self, batch, batch_nb, opt_idx, hiddens):
        """
        Handle forward for each training case (distributed, single gpu, etc...)
        :param batch:
        :param batch_nb:
        :return:
        """
        # ---------------
        # FORWARD
        # ---------------
        # enable not needing to add opt_idx to training_step
        args = [batch, batch_nb]
        if len(self.optimizers) > 1:
            args.append(opt_idx)

        # pass hiddens if using tbptt
        if self.truncated_bptt_steps is not None:
            args.append(hiddens)

        # distributed forward
        if self.use_ddp or self.use_ddp2 or self.use_dp:
            output = self.model(*args)

        # single GPU forward
        elif self.single_gpu:
            gpu_id = 0
            if type(self.data_parallel_device_ids) is list:
                gpu_id = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(batch, gpu_id)
            args[0] = batch
            output = self.model.training_step(*args)

        # CPU forward
        else:
            output = self.model.training_step(*args)

        # allow any mode to define training_end
        if self.is_overriden('training_end'):
            model_ref = self.get_model()
            output = model_ref.training_end(output)

        # format and reduce outputs accordingly
        output = self.process_output(output, train=True)

        return output
