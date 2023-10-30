from models import NamePostConfig  # 2. Import your custom model.

from lightning.app import LightningFlow, LightningApp
from lightning.app.api import Post


class Flow(LightningFlow):
    # 1. Define the state
    def __init__(self):
        super().__init__()
        self.names = []

    # 2. Optional, but used to validate names
    def run(self):
        print(self.names)

    # 3. Annotate your input with your custom pydantic model.
    def handle_post(self, config: NamePostConfig):
        self.names.append(config.name)
        return f'The name {config} was registered'

    # 4. Defines this Component's Restful API. You can have several routes.
    def configure_api(self):
        return [
            Post(
                route="/name",
                method=self.handle_post,
            )
        ]


app = LightningApp(Flow())
