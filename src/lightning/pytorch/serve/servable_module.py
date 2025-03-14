from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
from torch import Tensor


class ServableModule(ABC, torch.nn.Module):
    """The ServableModule provides a simple API to make your model servable.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Here is an example of how to use the ``ServableModule`` module.

    .. code-block:: python

        from typing import Dict, Any, Callable

        import torch

        from lightning.pytorch import Trainer
        from lightning.pytorch.demos.boring_classes import BoringModel
        from lightning.pytorch.serve.servable_module_validator import ServableModule, ServableModuleValidator


        class ServableBoringModel(BoringModel, ServableModule):
            def configure_payload(self) -> Dict[str, Any]:
                return {"body": {"x": list(range(32))}}

            def configure_serialization(self) -> Tuple[Dict[str, Callable], Dict[str, Callable]]:
                def deserialize(x):
                    return torch.tensor(x, dtype=torch.float)

                def serialize(x):
                    return x.tolist()

                return {"x": deserialize}, {"output": serialize}

            def serve_step(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"output": torch.tensor([0, 1])}

            def configure_response(self):
                return {"output": [0, 1]}


        serve_cb = ServableModuleValidator()
        trainer = Trainer(
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=0,
            callbacks=[serve_cb],
        )
        trainer.fit(ServableBoringModel())
        assert serve_cb.resp.json() == {"output": [0, 1]}

    """

    @abstractmethod
    def configure_payload(self) -> dict[str, Any]:
        """Returns a request payload as a dictionary."""

    @abstractmethod
    def configure_serialization(self) -> tuple[dict[str, Callable], dict[str, Callable]]:
        """Returns a tuple of dictionaries.

        The first dictionary contains the name of the ``serve_step`` input variables name as its keys
        and the associated de-serialization function (e.g function to convert a payload to tensors).

        The second dictionary contains the name of the ``serve_step`` output variables name as its keys
        and the associated serialization function (e.g function to convert a tensors into payload).

        """

    @abstractmethod
    def serve_step(self, *args: Tensor, **kwargs: Tensor) -> dict[str, Tensor]:
        r"""Returns the predictions of your model as a dictionary.

        .. code-block:: python

            def serve_step(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"predictions": self(x)}

        Args:
            args: The output from de-serializer functions provided by the ``configure_serialization`` hook.
            kwargs: The keyword output of the de-serializer functions provided by the ``configure_serialization`` hook.

        Return:
            - ``dict`` - A dictionary with their associated tensors.

        """

    @abstractmethod
    def configure_response(self) -> dict[str, Any]:
        """Returns a response to validate the server response."""
