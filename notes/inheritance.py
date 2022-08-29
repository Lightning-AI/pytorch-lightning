

class LiteStrategy:

    def setup_model_and_optimizers(self):
        pass

    def optimizer_step(self, module):
        pass


class LiteParallelStrategy(LiteStrategy):

    def distributed_sampler_kwargs(self):
        pass


class LitDDPStrategy(LiteParallelStrategy):
    pass


class PLStrategy:

    def lightning_module(self):
        pass

    def setup(self, trainer):
        pass


class PLParallelStrategy(PLStrategy):

    def distributed_sampler_kwargs(self):
        pass




