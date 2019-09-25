import torch


class ModelHooks(torch.nn.Module):

    def on_sanity_check_start(self):
        """
        Called before starting evaluate
        :return:
        """
        pass

    def on_batch_start(self, batch):
        pass

    def on_batch_end(self):
        pass

    def on_epoch_start(self):
        pass

    def on_epoch_end(self):
        pass

    def on_pre_performance_check(self):
        pass

    def on_post_performance_check(self):
        pass

    def on_training_metrics(self, metrics):
        pass

    def on_before_zero_grad(self, optimizer):
        """
        Called after optimizer.step() and before optimizer.zero_grad()

        for optimizer in optimizers:
            optimizer.step()
            model.on_before_zero_grad(optimizer) # < ---- called here
            optimizer.zero_grad

        :param optimizer:
        :return:
        """
        pass

    def on_after_backward(self):
        """
        Called after loss.backward() and before optimizers do anything
        :return:
        """
        pass
