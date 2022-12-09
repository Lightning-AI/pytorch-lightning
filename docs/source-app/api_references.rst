:orphan:

##############################
Lightning App - API References
##############################

Core
____

.. currentmodule:: lightning_app.core

.. autosummary::
    :toctree: api/
    :nosignatures:
    :template: classtemplate_no_index.rst

    LightningApp
    LightningFlow
    LightningWork

Learn more about :ref:`Lightning Core <core_api>`.

----

Built-in Components
___________________

.. currentmodule:: lightning_app.components

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate_no_index.rst

    ~database.client.DatabaseClient
    ~database.server.Database
    ~python.popen.PopenPythonScript
    ~python.tracer.TracerPythonScript
    ~training.LightningTrainerScript
    ~serve.gradio.ServeGradio
    ~serve.serve.ModelInferenceAPI
    ~serve.python_server.PythonServer
    ~serve.streamlit.ServeStreamlit
    ~multi_node.base.MultiNode
    ~multi_node.lite.LiteMultiNode
    ~multi_node.pytorch_spawn.PyTorchSpawnMultiNode
    ~multi_node.trainer.LightningTrainerMultiNode
    ~auto_scaler.AutoScaler

----

Frontend's
__________

.. currentmodule:: lightning_app.frontend

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate_no_index.rst

    ~frontend.Frontend
    ~web.StaticWebFrontend
    ~stream_lit.StreamlitFrontend

Learn more about :ref:`Frontend's <ui_and_frontends>`.

----

Storage
_______

.. currentmodule:: lightning_app.storage

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate_no_index.rst

    ~path.Path
    ~drive.Drive
    ~payload.Payload
    ~mount.Mount

Learn more about :ref:`Storage <storage>`.

----

Runners
_______

.. currentmodule:: lightning_app.runners

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate_no_index.rst

    ~cloud.CloudRuntime
    ~multiprocess.MultiProcessRuntime

----

lightning_app.utilities.packaging
_________________________________

.. currentmodule:: lightning_app.utilities.packaging

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate_no_index.rst

    ~cloud_compute.CloudCompute
    ~build_config.BuildConfig
