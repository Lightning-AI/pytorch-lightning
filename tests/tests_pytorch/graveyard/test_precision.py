

def test_backward_compatible_precision_plugin_imports():
    from lightning.pytorch.plugins.precision.precision_plugin import PrecisionPlugin
    from lightning.pytorch.plugins.precision.precision import PrecisionPlugin as PrecisionPluginNew
    assert PrecisionPluginNew is PrecisionPlugin
