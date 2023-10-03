import json
import subprocess

from lightning import BuildConfig, LightningWork
from lightning.app.storage.path import Path

# ML_SERVER_URL = https://github.com/SeldonIO/MLServer


class MLServer(LightningWork):
    """This components uses SeldonIO MLServer library.

    The model endpoint: /v2/models/{MODEL_NAME}/versions/{VERSION}/infer.

    Arguments:
        name: The name of the model for the endpoint.
        implementation: The model loader class.
            Example: "mlserver_sklearn.SKLearnModel".
            Learn more here: $ML_SERVER_URL/tree/master/runtimes
        workers: Number of server worker.

    """

    def __init__(
        self,
        name: str,
        implementation: str,
        workers: int = 1,
        **kwargs,
    ):
        super().__init__(
            parallel=True,
            cloud_build_config=BuildConfig(
                requirements=["mlserver", "mlserver-sklearn"],
            ),
            **kwargs,
        )
        # 1: Collect the config's.
        self.settings = {
            "debug": True,
            "parallel_workers": workers,
        }
        self.model_settings = {
            "name": name,
            "implementation": implementation,
        }
        # 2: Keep track of latest version
        self.version = 1

    def run(self, model_path: Path):
        """The model is downloaded when the run method is invoked.

        Arguments:
            model_path: The path to the trained model.

        """
        # 1: Use the host and port at runtime so it works in the cloud.
        # $ML_SERVER_URL/blob/master/mlserver/settings.py#L50
        if self.version == 1:
            # TODO: Reload the next version model of the model.

            self.settings.update({"host": self.host, "http_port": self.port})

            with open("settings.json", "w") as f:
                json.dump(self.settings, f)

            # 2. Store the model-settings
            # $ML_SERVER_URL/blob/master/mlserver/settings.py#L120
            self.model_settings["parameters"] = {
                "version": f"v0.0.{self.version}",
                "uri": str(model_path.absolute()),
            }
            with open("model-settings.json", "w") as f:
                json.dump(self.model_settings, f)

            # 3. Launch the Model Server
            subprocess.Popen("mlserver start .", shell=True)

            # 4. Increment the version for the next time run is called.
            self.version += 1

        else:
            # TODO: Load the next model and unload the previous one.
            pass

    def alive(self):
        # Current hack, when the url is available,
        # the server is up and running.
        # This would be cleaned out and automated.
        return self.url != ""
