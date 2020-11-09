from torch import optim


def reinit_scheduler_properties(self, optimizers: list, schedulers: list):
        # Reinitialize optimizer.step properties added by schedulers
        for scheduler in schedulers:
            scheduler = scheduler['scheduler']

            for optimizer in optimizers:
                state = None
                idx = 0

                # check that we dont mix users optimizers and schedulers
                if scheduler.optimizer == optimizer:
                    # Find the mro belonging to the base lr scheduler class
                    for i, mro in enumerate(scheduler.__class__.__mro__):
                        if mro in (optim.lr_scheduler._LRScheduler, optim.lr_scheduler.ReduceLROnPlateau):
                            idx = i
                            state = scheduler.state_dict()
                        else:
                            state = None

                scheduler.__class__.__mro__[idx].__init__(scheduler, optimizer)
                if state is not None:
                    scheduler.load_state_dict(state)