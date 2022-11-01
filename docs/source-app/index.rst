.. lightning documentation master file, created by
   sphinx-quickstart on Sat Sep 19 16:37:02 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

######################
Welcome to ⚡ Lightning
######################

.. join_slack::
   :align: left

Build models and full stack AI apps ⚡ *Lightning fast* (like these):

|

.. raw:: html

   <div class="display-card-container">
      <div class="row" style="display:flex; align-items: center; justify-content: center; gap: 10px">

.. app_card::
   :title: Train: LLM
   :width: 300
   :image: https://media3.giphy.com/media/KgEzIaqjorVRVGSvpU/giphy.gif
   :preview: https://lightning.ai
   :deploy: https://lightning.ai
   :target: https://apple.com
   :tags: Model

.. app_card::
   :title: Deploy: Difussion model
   :width: 300
   :image: https://media3.giphy.com/media/KgEzIaqjorVRVGSvpU/giphy.gif
   :preview: https://lightning.ai
   :deploy: https://lightning.ai
   :target: https://apple.com
   :tags: App

.. app_card::
   :title: Launch: Full stack AI app
   :width: 300
   :image: https://media3.giphy.com/media/KgEzIaqjorVRVGSvpU/giphy.gif
   :preview: https://lightning.ai
   :deploy: https://lightning.ai
   :target: https://apple.com
   :tags: App

.. raw:: html

      </div>
   </div>

----

************************
Build modular components
************************

.. raw:: html

   <div class="row">
      <div class='col-md-5' style="padding-top: 18px">
         <p>
            Build modular, self-contained Lightning components that can do anything from 
            train models, and deploy models to run web user interfaces. A Lightning component 
            manages its own infrastructure, cloud costs, networking and more. Connect components
            to form a full stack AI app that we call a <i>Lightning App</i>.
         </p>
         <p>Image [] -> [] -> []</p>
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

   Start from component templates <https://lightning.ai/components>
   Start from app templates <https://lightning.ai/apps>
   Examples <examples/index>
   Glossary <glossary/index>
   How to... <workflows/index>