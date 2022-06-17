from typing import Union

from lightning_app import LightningFlow, LightningWork
from lightning_app.structures import Dict, List

Component = Union[LightningFlow, LightningWork, Dict, List]
ComponentTuple = (LightningFlow, LightningWork, Dict, List)
