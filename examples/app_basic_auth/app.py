from lightning import LightningApp, LightningFlow
from lightning_app.api import Post
from lightning_app.api.user import User


class Flow(LightningFlow):
    def run(self):
        pass

    def handler(self, user: User):
        return user.user_id

    def configure_api(self):
        return [Post("/v1/auth/", self.handler)]


app = LightningApp(Flow())
