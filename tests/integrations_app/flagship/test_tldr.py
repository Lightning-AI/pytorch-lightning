import os

from lightning.testing import run_app_in_cloud
from integrations_app.flagship import _PATH_INTEGRATIONS_DIR

def test_app_in_cloud():

    with open(os.path.join(_PATH_INTEGRATIONS_DIR, 'test_app.py'), 'w') as f:
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

        expected_strings = [
            # don't include values for actual hardware availability as this may depend on environment.
            'GPU available: ',
            'All distributed processes registered.',
            '674 K    Trainable params\n0         Non - trainable params\n674 K    Total params\n2.699   Total estimated model params size(MB)',
            'Epoch 0:',
            '`Trainer.fit` stopped: `max_epochs=2` reached.',
            'Input text:Input text:\n summarize: ML Ops platforms come in many flavors from platforms that train models'
        ]

        with run_app_in_cloud(_PATH_INTEGRATIONS_DIR, 'test_app.py') as (_, _, fetch_logs, _):
            logs = fetch_logs()

        for curr_str in expected_strings:
            assert curr_str in logs
