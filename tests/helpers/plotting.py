import numpy as np

from pytorch_lightning.utilities import _module_available

_MATPLOTLIB_AVAILABLE = _module_available("matplotlib")
if _MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
else:

    class plt:
        figure = None


def dummy_figure() -> plt.figure:
    """Dummy figure to test logging capability of figures for loggers."""

    f = plt.figure()
    plt.plot(np.linspace(0., 1., 100), np.linspace(0., 10., 100)**2)

    return f
