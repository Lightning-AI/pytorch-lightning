.. lightning documentation master file, created by
   sphinx-quickstart on Sat Sep 19 16:37:02 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

######################
Welcome to ⚡ Lightning
######################
Build models and full stack AI apps ⚡ *Lightning fast*.

.. join_slack::
   :align: left

**Featured examples of what you can build with Lightning:**

|

.. raw:: html

   <div>
      <div class="row" style="display:flex; align-items: center; justify-content: center; gap: 10px">

.. app_card::
   :title: Develop and Train
   :description: Train an LLM (512 GPUs)
   :width: 280
   :image: https://lightning-ai-docs.s3.amazonaws.com/develop_n_train_v1.jpg
   :preview: https://lightning.ai
   :deploy: https://lightning.ai
   :target: https://apple.com
   :tags: Model

.. app_card::
   :title: Serve and deploy
   :description: Production-ready stable diffusion server (<2s latency)
   :width: 280
   :image: https://lightning-ai-docs.s3.amazonaws.com/serve_n_deploy_v1.jpg
   :preview: https://lightning.ai
   :deploy: https://lightning.ai
   :target: https://apple.com
   :tags: App

.. app_card::
   :title: Scale and build a product
   :description: Production-ready generative AI app
   :width: 280
   :image: https://lightning-ai-docs.s3.amazonaws.com/scale_n_build_v1.jpg
   :target: https://lightning.ai/muse
   :deploy: https://lightning.ai
   :tags: App

.. raw:: html

      </div>
   </div>

----

************************
Build modular components
************************

.. raw:: html

   <div style="display: flex; align-items: center; justify-content: center;">
      <img src="https://lightning-ai-docs.s3.amazonaws.com/intro_components.gif" style="max-width: 800px"></img>
   </div>

.. raw:: html

   <div class="row">
      <div class='col-md-5' style="padding-top: 18px">
         <p>
         Build modular, self-contained components that can train and deploy models, host a web UI or run arbitrary code on the cloud.
         A Lightning Component manages its own infrastructure, cloud costs, networking and more, so you can focus on application logic and not engineering.

         <br><br>
         Combine your components and prebuilt ones from <a href="https://lightning.ai/components" target="_blank">our gallery </a>
         to create full-stack AI apps, ⚡ <i>Lightning fast</i>.
         </p>
      </div>
      <div class='col-md-7'>

.. lit_tabs::
   :code_files: landing_app.py;
   :highlights: 6
   :app_id: abc123
   :height: 250px
   :code_only: True

.. lit_tabs::
   :code_files: landing_app_run.bash
   :highlights: 6
   :height: 150px
   :code_only: True

|

.. raw:: html

      </div>
   </div>

|

.. raw:: html

    <div class="display-card-container" style="padding: 0 20px 0 20px">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Get started
   :description: Learn to go from Lightning components to full stack AI apps step-by-step.
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

   Install <install/installation>
   self

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

   Start from component templates <https://lightning.ai/components>
   Start from app templates <https://lightning.ai/apps>
   Examples <examples/index>
   Glossary <glossary/index>
   How to... <workflows/index>
