from typing import Any, Callable, Dict, Tuple

import torch


class ServableModule(torch.nn.Module):

    """The ServableModule provides a simple API to make your model servable.

    .. warning::

        This is currently an experimental feature and API changes are to be expected.

    Here is an example on how to use the `ServableModule` module.

    .. code-block:: python

        from typing import Dict

        import torch

        from pytorch_lightning import seed_everything, Trainer
        from pytorch_lightning.demos.boring_classes import BoringModel
        from pytorch_lightning.serve.servable_module_validator import ServableModule, ServableModuleValidator


        class ServableBoringModel(BoringModel, ServableModule):
            def configure_payload(self) -> ...:
                return {"body": {"x": list(range(32))}}

            def configure_serialization(self):
                def deserialize(x):
                    return torch.tensor(x, dtype=torch.float)

                def serialize(x):
                    return x.tolist()

                return {"x": deserialize}, {"output": serialize}

            def serve_step(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                assert torch.equal(x, torch.arange(32, dtype=torch.float))
                return {"output": torch.tensor([0, 1])}


        callback = ServableModuleValidator()
        trainer = Trainer(
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=0,
            callbacks=[callback],
        )
        trainer.fit(ServableBoringModel())
        assert callback.resp.json() == {"output": [0, 1]}
    """

    def configure_payload(self) -> Dict[str, Any]:
        """Returns a request payload as a dictionary."""
        ...

    def configure_serialization(self) -> Tuple[Dict[str, Callable], Dict[str, Callable]]:
        """Returns a tuple of dictionaries.

        The first dictionary contains the name of the ``serve_step`` input variables name as its keys
        and the associated de-serialization function (e.g function to convert a payload to tensors).

        The second dictionary contains the name of the ``serve_step`` output variables name as its keys
        and the associated serialization function (e.g function to convert a tensors into payload).
        """
        ...

    def serve_step(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""
        Returns the predictions of your model as a dictionary.

        .. code-block:: python

            def serve_step(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"predictions": self(x)}

        Args:
            args|kwargs (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                The output of the serializer functions provided by the ``configure_serialization`` hook.

        Return:
            - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``
        """
        ...

    def configure_response(self) -> Dict[str, Any]:
        """Returns a response to validate the server response."""
        ...
