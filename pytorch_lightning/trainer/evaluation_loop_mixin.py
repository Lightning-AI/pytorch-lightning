import torch

from pytorch_lightning.utilities.debugging import MisconfigurationException


class TrainerEvaluationLoopMixin(object):

    def evaluate(self, model, dataloaders, max_batches, test=False):
        """
        Run evaluation code
        :param model: PT model
        :param dataloaders: list of PT dataloaders
        :param max_batches: Scalar
        :param test: boolean
        :return:
        """
        # enable eval mode
        model.zero_grad()
        model.eval()

        # copy properties for forward overrides
        self.copy_trainer_model_properties(model)

        # disable gradients to save memory
        torch.set_grad_enabled(False)

        # bookkeeping
        outputs = []

        # run training
        for dataloader_idx, dataloader in enumerate(dataloaders):
            dl_outputs = []
            for batch_idx, batch in enumerate(dataloader):

                if batch is None:  # pragma: no cover
                    continue

                # stop short when on fast_dev_run (sets max_batch=1)
                if batch_idx >= max_batches:
                    break

                # -----------------
                # RUN EVALUATION STEP
                # -----------------
                output = self.evaluation_forward(model,
                                                 batch,
                                                 batch_idx,
                                                 dataloader_idx,
                                                 test)

                # track outputs for collation
                dl_outputs.append(output)

                # batch done
                if self.show_progress_bar:
                    self.progress_bar.update(1)
            outputs.append(dl_outputs)

        eval_results = {}

        # with a single dataloader don't pass an array
        if len(dataloaders) == 1:
            outputs = outputs[0]

        # give model a chance to do something with the outputs (and method defined)
        model = self.get_model()
        if test and self.is_overriden('test_end'):
            eval_results = model.test_end(outputs)
        elif self.is_overriden('validation_end'):
            eval_results = model.validation_end(outputs)

        # enable train mode again
        model.train()

        # enable gradients to save memory
        torch.set_grad_enabled(True)

        return eval_results

    def run_evaluation(self, test=False):
        # when testing make sure user defined a test step
        can_run_test_step = False
        if test:
            can_run_test_step = self.is_overriden('test_step') and self.is_overriden('test_end')
            if not can_run_test_step:
                m = '''You called .test() without defining a test step or test_end.
                Please define and try again'''
                raise MisconfigurationException(m)

        # validate only if model has validation_step defined
        # test only if test_step or validation_step are defined
        run_val_step = self.is_overriden('validation_step')

        if run_val_step or can_run_test_step:

            # hook
            model = self.get_model()
            model.on_pre_performance_check()

            # select dataloaders
            if test:
                dataloaders = self.get_test_dataloaders()
                max_batches = self.nb_test_batches
            else:
                # val
                dataloaders = self.get_val_dataloaders()
                max_batches = self.nb_val_batches

            # cap max batches to 1 when using fast_dev_run
            if self.fast_dev_run:
                max_batches = 1

            # run evaluation
            eval_results = self.evaluate(self.model,
                                         dataloaders,
                                         max_batches,
                                         test)
            _, prog_bar_metrics, log_metrics, callback_metrics = self.process_output(eval_results)

            # add metrics to prog bar
            self.add_tqdm_metrics(prog_bar_metrics)

            # log metrics
            self.log_metrics(log_metrics, {})

            # track metrics for callbacks
            self.callback_metrics = callback_metrics

            # hook
            model.on_post_performance_check()

            if self.show_progress_bar:
                # add model specific metrics
                tqdm_metrics = self.training_tqdm_dict
                self.progress_bar.set_postfix(**tqdm_metrics)

        # model checkpointing
        if self.proc_rank == 0 and self.checkpoint_callback is not None and not test:
            self.checkpoint_callback.on_epoch_end(epoch=self.current_epoch,
                                                  logs=self.callback_metrics)

    def evaluation_forward(self, model, batch, batch_idx, dataloader_idx, test=False):
        # make dataloader_idx arg in validation_step optional
        args = [batch, batch_idx]

        if test and len(self.get_test_dataloaders()) > 1:
            args.append(dataloader_idx)

        elif not test and len(self.get_val_dataloaders()) > 1:
            args.append(dataloader_idx)

        # handle DP, DDP forward
        if self.use_ddp or self.use_dp or self.use_ddp2:
            output = model(*args)
            return output

        # single GPU
        if self.single_gpu:
            # for single GPU put inputs on gpu manually
            root_gpu = 0
            if type(self.data_parallel_device_ids) is list:
                root_gpu = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(batch, root_gpu)
            args[0] = batch

        # CPU
        if test:
            output = model.test_step(*args)
        else:
            output = model.validation_step(*args)

        return output
