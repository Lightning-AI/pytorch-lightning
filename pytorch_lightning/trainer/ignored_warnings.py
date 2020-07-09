import warnings


def ignore_scalar_return_in_dp():
    # Users get confused by this warning so we silence it
    warnings.filterwarnings('ignore', message='Was asked to gather along dimension 0, but all '
                                              'input tensors were scalars; will instead unsqueeze '
                                              'and return a vector.')


ignore_scalar_return_in_dp()
