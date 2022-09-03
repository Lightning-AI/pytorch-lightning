####################################
Add a Label Studio annotator (basic)
####################################

**Audience:** Users who want to add a Label Studio annotator.

.. warning::

    The Component in this article is not in the official Lightning AI Component Gallery. It is an Early Access version of the Component.
	The repo for this Component is here: https://github.com/robert-s-lee/lit_label_studio. Refer to the Readme for more information.

----

*********************
What is Label Studio?
*********************

Label Studio is an open source data labeling tool.

----

**********************************
Install the Label Studio Component
**********************************



To install the Component use the following command:

.. code:: python

    python -m lightning install component git+https://github.com/robert-s-lee/lit_label_studio.git@0.0.0

To verify that the component is installed:

.. code:: python

    python -m pip show lit_label_studio

----

**********************
Setup lit_label_studio
**********************

Use Conda for Lightning and use ``venv`` for Label Studio. Label Studio and Lightning have library version conflict. ``venv`` is used in the Lightning Cloud.

.. code:: python

    virtualenv ~/venv-label-studio
    git clone https://github.com/robert-s-lee/label-studio; pushd label-studio; git checkout x-frame-options; popd source ~/venv-label-studio/bin/activate; pushd label-studio; which python; python -m pip install -e .; popd; deactivate


Test label-studio

.. code:: python

    export LABEL_STUDIO_X_FRAME_OPTIONS='allow-from *'
    source ~/venv-label-studio/bin/activate; cd label-studio; python label_studio/manage.py migrate; python label_studio/manage.py runserver; cd ..; deactivate

----

*****************************
Add the Component to your App
*****************************

Once the Component is installed, use this to include it in your App:

.. code:: python

    from lit_label_studio import LitLabelStudio

    import lightning_app as la

    class LitApp(la.LightningFlow):
        def __init__(self) -> None:
            super().__init__()
            self.lit_label_studio = LitLabelStudio()

        def run(self):
            self.lit_label_studio.run()

    app = la.LightningApp(LitApp())


***********
Run the App
***********
Run the App locally:

.. code:: python

    export LABEL_STUDIO_X_FRAME_OPTIONS='allow-from *'
    lightning run app app.py

Run the App in the cloud:

.. code:: python

    lightning run app app.py --cloud
