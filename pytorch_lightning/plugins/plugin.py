from pytorch_lightning.utilities import AMPType


class LightningPlugin:
    """
    Defines base class for Plugins. Plugins represent functionality that can be injected into the lightning codebase.
    """

    def required_plugins(self, amp_backend: AMPType, trainer) -> list:
        """
            Override to define additional required plugins. This is useful for when custom plugins
            need to enforce override of other plugins.

        Returns: Optional list of plugins containing additional plugins.

        Example::
            class MyPlugin(DDPPlugin):
                def required_plugins(self):
                    return [MyCustomAMPPlugin()]

            # Will automatically add the necessary AMP plugin
            trainer = Trainer(plugins=[MyPlugin()])

            # Crash as MyPlugin enforces custom AMP plugin
            trainer = Trainer(plugins=[MyPlugin(), NativeAMPPlugin()])

        """
        return []
