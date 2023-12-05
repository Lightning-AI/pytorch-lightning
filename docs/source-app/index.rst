.. lightning documentation master file, created by
   sphinx-quickstart on Sat Sep 19 16:37:02 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#######################
Welcome to ⚡ Lightning
#######################
Build models, ML components and full stack AI apps ⚡ *Lightning fast*.

**Featured examples of what you can do with Lightning:**

|

.. raw:: html

   <div>
      <div class="row" style="display:flex; align-items: center; justify-content: center; gap: 10px">

.. app_card::
   :title: Develop and Train
   :description: Train a model (32 GPUs)
   :width: 280
   :image: https://lightning-ai-docs.s3.amazonaws.com/develop_n_train_v1.jpg
   :target: levels/basic/real_lightning_component_implementations.html#ex-pytorch-lightning-trainer
   :preview: levels/basic/real_lightning_component_implementations.html#ex-pytorch-lightning-trainer
   :tags: Training

.. app_card::
   :title: Serve and deploy
   :description: Develop a Model Server
   :width: 280
   :image: https://lightning-ai-docs.s3.amazonaws.com/serve_n_deploy_v1.jpg
   :target: examples/model_server_app/model_server_app.html
   :preview: examples/model_server_app/model_server_app.html
   :tags: Serving

.. app_card::
   :title: Scale and build a product
   :description: Production-ready generative AI app
   :width: 280
   :app_id: HvUwbEG90E
   :image: https://lightning-ai-docs.s3.amazonaws.com/scale_n_build_v1.jpg
   :target: https://lightning.ai/app/HvUwbEG90E-Muse
   :tags: AI App

.. raw:: html

      </div>
   </div>

----

********************************
Build self-contained, components
********************************
Use Lightning, the hyper-minimalistic framework, to build machine learning components that can plug into existing ML workflows.
A Lightning component organizes arbitrary code to run on the cloud, manage its own infrastructure, cloud costs, networking, and more.
Focus on component logic and not engineering.

Use components on their own, or compose them into full-stack AI apps with our next-generation Lightning orchestrator.

.. raw:: html

   <div style="display: flex; align-items: center; justify-content: center;">
      <img src="https://lightning-ai-docs.s3.amazonaws.com/intro_components.gif" style="max-width: 800px"></img>
   </div>

|

|

**Run an example component on the cloud**:

.. include:: ./levels/basic/hero_components.rst

|

Components run the same on the cloud and locally on your choice of hardware.

.. lit_tabs::
   :code_files: landing_app_run.bash
   :highlights: 5
   :height: 150px
   :code_only: True

Explore pre-built community components in `our gallery <https://lightning.ai/components>`_.

|

.. raw:: html

    <div class="display-card-container" style="padding: 0 20px 0 20px">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Get started
   :description: Learn to build Lightning components step-by-step.
   :col_css: col-md-12
   :button_link: levels/basic/index.html
   :height: 160
   :tag: 10 minutes

.. raw:: html

        </div>
    </div>

.. raw:: html

   <div style="display:none">

.. toctree::
   :maxdepth: 1
   :caption: Home

   self
   Install <install/installation>

.. toctree::
   :maxdepth: 1
   :caption: Get started in steps

   Basic <levels/basic/index>
   Intermediate <levels/intermediate/index>
   Advanced <levels/advanced/index>

.. toctree::
   :maxdepth: 1
   :caption: Core API Reference

   LightningApp <core_api/lightning_app/index>
   LightningFlow <core_api/lightning_flow>
   LightningWork <core_api/lightning_work/index>

.. toctree::
   :maxdepth: 1
   :caption: Addons API Reference

   api_reference/components
   api_reference/frontend
   api_reference/runners
   api_reference/storage

.. toctree::
   :maxdepth: 1
   :caption: More

   Examples <examples/index>
   Glossary <glossary/index>
   How-to <workflows/index>
