########################
lightning.app.components
########################

.. contents::
    :depth: 1
    :local:
    :backlinks: top

.. currentmodule:: lightning.app.components


Built-in Components
___________________

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ~database.client.DatabaseClient
    ~database.server.Database
    ~python.popen.PopenPythonScript
    ~python.tracer.TracerPythonScript
    ~training.LightningTrainerScript
    ~serve.gradio_server.ServeGradio
    ~serve.serve.ModelInferenceAPI
    ~serve.python_server.PythonServer
    ~serve.streamlit.ServeStreamlit
    ~multi_node.base.MultiNode
    ~multi_node.fabric.FabricMultiNode
    ~multi_node.pytorch_spawn.PyTorchSpawnMultiNode
    ~multi_node.trainer.LightningTrainerMultiNode
    ~serve.auto_scaler.AutoScaler
    ~serve.auto_scaler.ColdStartProxy
