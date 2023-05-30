from lightning.app import LightningFlow, LightningWork


class MyCustomLightningWork(LightningWork):
    @staticmethod
    def special_method():
        return "Hi, I'm an external lightning work component and can be added to any lightning project."


class MyCustomLightningFlow(LightningFlow):
    @staticmethod
    def special_method():
        return "Hi, I'm an external lightning flow component and can be added to any lightning project."


def exported_lightning_components():
    return [MyCustomLightningWork, MyCustomLightningFlow]
