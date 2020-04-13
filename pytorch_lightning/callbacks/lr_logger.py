r"""



"""

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException

class LRLogger(Callback):
    def __init__(self):
        self.lrs = {}
        self.names = []
    
    def on_train_start(self, trainer, pl_module):
        if trainer.lr_schedulers == [ ]:
            raise MisconfigurationException(
                'Cannot use LRLogger callback with models that have no'
                ' learning rate schedulers.')

        # Create uniqe names in the case we have multiple of the same learning
        # rate schduler + multiple parameter groups
        names = [ ]
        for scheduler in trainer.lr_schedulers:
            sch = scheduler['scheduler']
            sch_name = sch.__class__.__name__
            name = sch_name
            counter = 0
            # Multiple schduler of the same type
            while True:
                counter += 1
                if name in names:
                    name = sch_name + '-' + counter
                else:
                    break
            
            # Multiple param groups for the same schduler
            param_groups = sch.optimizer.param_groups
            if len(param_groups) != 1:
                for i, pg in enumerate(param_groups):
                    temp = name + '/' + str(i+1)
                    names.append(temp)
            else:
                names.append(name)
            
            self.names.append(name)
                
        # Initialize for storing values
        for name in names:
            self.lrs[name] = []
    
    def on_batch_start(self, trainer, pl_module):
        latest_stat = self._extract_lr(trainer, 'step')
        if trainer.logger and latest_stat != {}:
            trainer.logger.log_metrics(latest_stat, step=trainer.global_step)
        
    def on_epoch_start(self, trainer, pl_module):
        latest_stat = self._extract_lr(trainer, 'epoch')
        if trainer.logger and latest_stat != {}:
            trainer.logger.log_metrics(latest_stat, step=trainer.global_step)
        
    def _extract_lr(self, trainer, interval):
        latest_stat = { }
        for name, scheduler in zip(self.names, trainer.lr_schedulers):
            if scheduler['interval'] == interval:
                param_groups = scheduler['scheduler'].optimizer.param_groups
                if len(param_groups) != 1:
                    for i, pg in enumerate(param_groups):
                        lr = pg['lr']
                        self.lrs[name + '/' + str(i+1)].append(lr)
                        latest_stat[name + '/' + str(i+1)] = lr
                else:
                    self.lrs[name].append(param_groups[0]['lr'])
                    latest_stat[name] = param_groups[0]['lr']
        return latest_stat
                    