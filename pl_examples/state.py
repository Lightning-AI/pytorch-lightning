# import torch
#
# state1 = torch.default_generator.get_state()
#
#
# torch.rand(2,2)
#
# state2 = torch.default_generator.get_state()
# assert torch.equal(state1, state2)
#


class Custom:

    var = 1
    pass

print(Custom().__dict__)