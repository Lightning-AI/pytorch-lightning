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

    # 3. Method executed when a request is received.
    def handle_post(self, name: str):
        self.names.append(name)
        return f'The name {name} was registered'

    # 4. Defines this Component's Restful API. You can have several routes.
    def configure_api(self):
        # Your own defined route and handler
        return [Post(route="/name", method=self.handle_post)]


app = LightningApp(Flow())
