from typing import Any, Type

from lightning_app import structures
from lightning_app.core.flow import LightningFlow
from lightning_app.core.work import LightningWork
from lightning_app.utilities.enum import WorkStageStatus
from lightning_app.utilities.packaging.cloud_compute import CloudCompute


class ContinuousDeployment(LightningFlow):
    def __init__(
        self,
        train_component,
        deploy_component,
        *args,
        **kwargs
    ) -> None:
        """This component enables continuous deployment

        Example::

            TODO

        Arguments:
            train_component: ABC
            deploy_component: DEF
        """
        super().__init__()
        self.ws = structures.List()
        self.train_component = train_component
        self.deploy_component = deploy_component

    def run(self) -> None:
        self.train_component.run()
        self.deploy_component.run(self.train.redeploy_key)
