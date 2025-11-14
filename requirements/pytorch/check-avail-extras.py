import sys
import warnings

if __name__ == "__main__":
    if sys.platform == "win32":
        # ignore warnings related to Python 3.13 and Numpy incompatibility on Windows
        numpy_warnings = [
            r"invalid value encountered in exp2.*",
            r"invalid value encountered in nextafter.*",
            r"invalid value encountered in log10.*",
        ]

        for w in numpy_warnings:
            warnings.filterwarnings(
                action="ignore",
                message=w,
                category=RuntimeWarning,
            )

    import hydra  # noqa: F401
    import jsonargparse  # noqa: F401
    import matplotlib  # noqa: F401
    import omegaconf  # noqa: F401
    import rich  # noqa: F401
