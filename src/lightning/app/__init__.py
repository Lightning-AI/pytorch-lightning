import sys

from lightning_utilities.core.imports import RequirementCache, module_available

__all__ = []

if not RequirementCache("lightning_app"):
    raise ModuleNotFoundError("Please, run `pip install lightning-app`")  # E111

else:
    import litdata

    # Enable resolution at least for lower data namespace
    sys.modules["lightning.app"] = lightning_app

    from lightning_app.core.app import LightningApp  # noqa: E402
    from lightning_app.core.flow import LightningFlow  # noqa: E402
    from lightning_app.core.work import LightningWork  # noqa: E402
    from lightning_app.plugin.plugin import LightningPlugin  # noqa: E402
    from lightning_app.utilities.packaging.build_config import BuildConfig  # noqa: E402
    from lightning_app.utilities.packaging.cloud_compute import CloudCompute  # noqa: E402

    if module_available("lightning_app.components.demo"):
        from lightning.app.components import demo  # noqa: F401

    __all__ = ["LightningApp", "LightningFlow", "LightningWork", "LightningPlugin", "BuildConfig", "CloudCompute"]
