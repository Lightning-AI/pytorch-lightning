import torch

from pytorch_lightning.root_module import memory


class TrainerLoggingMixin(object):

    def log_metrics(self, metrics, grad_norm_dic):
        """
        Logs the metric dict passed in
        :param metrics:
        :param grad_norm_dic:
        :return:
        """
        # added metrics by Lightning for convenience
        metrics['epoch'] = self.current_epoch

        # add gpu memory
        if self.on_gpu and self.log_gpu_memory:
            mem_map = memory.get_memory_profile(self.log_gpu_memory)
            metrics.update(mem_map)

        # add norms
        metrics.update(grad_norm_dic)

        # turn all tensors to scalars
        scalar_metrics = self.metrics_to_scalars(metrics)

        # log actual metrics
        if self.proc_rank == 0 and self.logger is not None:
            self.logger.log_metrics(scalar_metrics, step_num=self.global_step)
            self.logger.save()

    def add_tqdm_metrics(self, metrics):
        for k, v in metrics.items():
            if type(v) is torch.Tensor:
                v = v.item()

            self.tqdm_metrics[k] = v

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is dict:
                v = self.metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics

    def process_output(self, output, train=False):
        """
        Reduces output according to the training mode.
        Separates loss from logging and tqdm metrics
        :param output:
        :return:
        """
        # ---------------
        # EXTRACT CALLBACK KEYS
        # ---------------
        # all keys not progress_bar or log are candidates for callbacks
        callback_metrics = {}
        for k, v in output.items():
            if k not in ['progress_bar', 'log']:
                callback_metrics[k] = v

        if train and (self.use_dp or self.use_ddp2):
            nb_gpus = self.num_gpus
            callback_metrics = self.reduce_distributed_output(callback_metrics, nb_gpus)

        for k, v in callback_metrics.items():
            callback_metrics[k] = v.item()

        # ---------------
        # EXTRACT PROGRESS BAR KEYS
        # ---------------
        try:
            progress_output = output['progress_bar']

            # reduce progress metrics for tqdm when using dp
            if train and (self.use_dp or self.use_ddp2):
                nb_gpus = self.num_gpus
                progress_output = self.reduce_distributed_output(progress_output, nb_gpus)

            progress_bar_metrics = progress_output
        except Exception:
            progress_bar_metrics = {}

        # ---------------
        # EXTRACT LOGGING KEYS
        # ---------------
        # extract metrics to log to experiment
        try:
            log_output = output['log']

            # reduce progress metrics for tqdm when using dp
            if train and (self.use_dp or self.use_ddp2):
                nb_gpus = self.num_gpus
                log_output = self.reduce_distributed_output(log_output, nb_gpus)

            log_metrics = log_output
        except Exception:
            log_metrics = {}

        # ---------------
        # EXTRACT LOSS
        # ---------------
        # if output dict doesn't have the keyword loss
        # then assume the output=loss if scalar
        loss = None
        if train:
            try:
                loss = output['loss']
            except Exception:
                if type(output) is torch.Tensor:
                    loss = output
                else:
                    raise RuntimeError(
                        'No `loss` value in the dictionary returned from `model.training_step()`.'
                    )

            # when using dp need to reduce the loss
            if self.use_dp or self.use_ddp2:
                loss = self.reduce_distributed_output(loss, self.num_gpus)

        # use every metric passed in as a candidate for callback
        callback_metrics.update(progress_bar_metrics)
        callback_metrics.update(log_metrics)

        # convert tensors to numpy
        for k, v in callback_metrics.items():
            if isinstance(v, torch.Tensor):
                callback_metrics[k] = v.item()

        return loss, progress_bar_metrics, log_metrics, callback_metrics

    def reduce_distributed_output(self, output, nb_gpus):
        if nb_gpus <= 1:
            return output

        # when using DP, we get one output per gpu
        # average outputs and return
        if type(output) is torch.Tensor:
            return output.mean()

        for k, v in output.items():
            # recurse on nested dics
            if isinstance(output[k], dict):
                output[k] = self.reduce_distributed_output(output[k], nb_gpus)

            # reduce only metrics that have the same nb of gpus
            elif output[k].size(0) == nb_gpus:
                reduced = torch.mean(output[k])
                output[k] = reduced
        return output
