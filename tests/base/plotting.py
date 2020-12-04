import numpy as np
from matplotlib import pyplot as plt


def dummy_figure():
    """Dummy figure to test logging capability of figures for loggers."""
    f = plt.figure()
    plt.plot(np.linspace(0., 1., 100), np.linspace(0., 10., 100) ** 2)

    return f
