
class StrategyBase:
    def setup_model_and_optimizers(self):
        pass

    def optimizer_step(self, module, *args, **kwargs):
        pass

class LiteStrategy(StrategyBase):
    pass



class LiteParallelStrategy(LiteStrategy):

    def distributed_sampler_kwargs(self):
        pass


class LiteDDPStrategy(LiteParallelStrategy):
    pass


class PLStrategy(StrategyBase):

    def lightning_module(self):
        pass

    def setup(self, trainer):
        pass

    def optimizer_step(self, module):
        pass


class PLParallelStrategy(PLStrategy):

    def distributed_sampler_kwargs(self):
        pass


class PLDDPStrategy(PLParallelStrategy):

    lite_ddp_strategy = LiteDDPStrategy(..)

    def distributed_sampler_kwargs(self):
        self.lite_ddp_strategy.distributed_sampler_kwargs()


