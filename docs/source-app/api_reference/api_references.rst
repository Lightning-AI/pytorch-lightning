:orphan:

##############################
Lightning App - API References
##############################

Core
----

.. currentmodule:: lightning_app.core

.. autosummary::
    :toctree: api
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

    ~serve.serve.ModelInferenceAPI
    ~python.popen.PopenPythonScript
    ~serve.gradio.ServeGradio
    ~python.tracer.TracerPythonScript

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

    ~drive.Drive
    ~path.Path
    ~payload.Payload

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
    ~singleprocess.SingleProcessRuntime
