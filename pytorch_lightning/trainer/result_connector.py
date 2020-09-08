import torch
from pytorch_lightning.utilities import flatten_dict


class ResultConnector:

    def __init__(self, trainer):
        self.trainer = trainer

    def log_evaluation_epoch_metrics(self, eval_results, using_eval_result):
        if using_eval_result:
            if isinstance(eval_results, list):
                for eval_result in eval_results:
                    self.trainer.logger_connector.callback_metrics = eval_result.callback_metrics
            else:
                self.trainer.logger_connector.callback_metrics = eval_results.callback_metrics
        else:
            if isinstance(eval_results, list):
                for eval_result in eval_results:
                    # with a scalar return, auto set it to "val_loss" for callbacks
                    if isinstance(eval_result, torch.Tensor):
                        flat = {'val_loss': eval_result}
                    else:
                        flat = flatten_dict(eval_result)
                    self.trainer.logger_connector.callback_metrics.update(flat)
            else:
                # with a scalar return, auto set it to "val_loss" for callbacks
                if isinstance(eval_results, torch.Tensor):
                    flat = {'val_loss': eval_results}
                else:
                    flat = flatten_dict(eval_results)
                self.trainer.logger_connector.callback_metrics.update(flat)
