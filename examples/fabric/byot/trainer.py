from typing import Union, List, Optional, Any
from lightning.fabric import Fabric, Accelerator, Strategy, _PRECISION_INPUT, _PLUGIN_INPUT, Logger

class Trainer:
    def __init__(self, 
                 accelerator: Union[str, Accelerator] = "auto",

        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        num_nodes: int = 1,
        precision: _PRECISION_INPUT = "32-true",
        plugins: Optional[Union[_PLUGIN_INPUT, List[_PLUGIN_INPUT]]] = None,
        callbacks: Optional[Union[List[Any], Any]] = None,
        loggers: Optional[Union[Logger, List[Logger]]] = None,
    ) -> None:

        self.fabric = Fabric(accelerator=accelerator, strategy=strategy, devices=devices, precision=precision, plugins=plugins, callbacks=callbacks, loggers=loggers)
