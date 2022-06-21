import os
from typing import Dict, List, Optional, TYPE_CHECKING

from lightning_app import LightningFlow

if TYPE_CHECKING:
    import wandb


class WeightsAndBiases(LightningFlow):
    def __init__(self, username: str, project_name: str, run_id: str, api_key: Optional[str] = None) -> None:
        super().__init__()
        self.username = username
        self.project_name = project_name
        self.run_id = run_id
        self._api_key = api_key
        self._run: Optional[wandb.Run] = None

    def run(self) -> None:
        if self._run is not None:
            return

        if self._api_key:
            os.environ["WANDB_API_KEY"] = self._api_key

        import wandb

        self._run = wandb.init(project=self.project_name, id=self.run_id, entity=self.username)

    def configure_layout(self) -> List[Dict[str, str]]:
        if self._run is not None:
            return [{"name": "Training Logs", "content": self._run.get_url()}]
        return []
