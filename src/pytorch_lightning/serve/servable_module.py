from typing import Any, Callable, Dict, Tuple

import torch


class ServableModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def configure_payload(self) -> Dict[str, Any]:
        ...

    def configure_serialization(self) -> Tuple[Dict[str, Callable], Dict[str, Callable]]:
        ...

    def serve_step(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        ...
