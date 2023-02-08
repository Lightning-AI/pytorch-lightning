import os
from time import sleep

from integrations_app.flagship import _PATH_INTEGRATIONS_DIR

from lightning.app.testing.testing import run_app_in_cloud
from lightning.app.utilities.imports import _is_playwright_available

if _is_playwright_available():
    import playwright


def test_app_in_cloud():

    with open(os.path.join(_PATH_INTEGRATIONS_DIR, "test_app.py"), "w") as f:
        app_string = """
        import lightning
        from tests.test_app import DummyTLDR
        app = lightning.app.LightningApp(
            MultiNodeLightningTrainerWithTensorboard(
                DummyTLDR, num_nodes=2, cloud_compute=lightning.CloudCompute("gpu-fast-multi", disk_size=50),
            )
        )
        """
        f.write(app_string)

        with run_app_in_cloud(_PATH_INTEGRATIONS_DIR, "test_app.py") as (
            _,
            view_page,
            fetch_logs,
            name,
        ):

            # def check_training_finished(*_, **__):
            #     locator = view_page.frame_locator("iframe").locator(
            #         'ul:has-text("`Trainer.fit` stopped: `max_epochs=2` reached.")'
            #     )
            #     if len(locator.all_text_contents()):
            #         return True
            #
            # wait_for(view_page, check_training_finished)

            logs = []
            while True:
                sleep(10)
                try:
                    logs = list(fetch_logs("multinode.ws.0"))
                    if any(["`Trainer.fit` stopped: `max_epochs=2` reached." in line for line in logs]):
                        break
                except (
                    playwright._impl._api_types.Error,
                    playwright._impl._api_types.TimeoutError,
                ):
                    pass

        expected_strings = [
            # don't include values for actual hardware availability as this may depend on environment.
            "GPU available: ",
            "All distributed processes registered.",
            "674 K    Trainable params\n0         Non - trainable params\n674 K    Total params\n2.699   Total estimated model params size(MB)",
            "Epoch 0:",
            "`Trainer.fit` stopped: `max_epochs=2` reached.",
            "Input text:Input text:\n summarize: ML Ops platforms come in many flavors from platforms that train models",
        ]

        for curr_str in expected_strings:
            assert any([curr_str in line for line in logs])
        os.remove(os.path.join(_PATH_INTEGRATIONS_DIR, "test_app.py"))
