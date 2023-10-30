import os
from unittest import mock
from unittest.mock import ANY


@mock.patch.dict(os.environ, {"LIGHTING_TESTING": "1"})
@mock.patch("lightning.app.components.serve.gradio_server.gradio")
def test_serve_gradio(gradio_mock):
    from lightning.app.components.serve.gradio_server import ServeGradio

    class MyGradioServe(ServeGradio):
        inputs = gradio_mock.inputs.Image(type="pil")
        outputs = gradio_mock.outputs.Image(type="pil")
        examples = [["./examples/app/components/serve/gradio/beyonce.png"]]

        def build_model(self):
            super().build_model()
            return "model"

        def predict(self, *args, **kwargs):
            super().predict(*args, **kwargs)
            return "prediction"

    comp = MyGradioServe()
    comp.run()
    assert comp.model == "model"
    assert comp.predict() == "prediction"
    gradio_mock.Interface.assert_called_once_with(
        fn=ANY, inputs=ANY, outputs=ANY, examples=ANY, title=None, description=None, theme=ANY
    )
