.. lightning documentation master file, created by
   sphinx-quickstart on Sat Sep 19 16:37:02 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

######################
Welcome to ⚡ Lightning
######################

Build models and full stack AI apps ⚡ *Lightning fast*

.. join_slack::
   :align: left

***************************************************
With Lightning, build full stack AI apps like these
***************************************************

|

.. raw:: html

   <div class="display-card-container">
      <div class="row" style="display:flex; align-items: center; justify-content: center; gap: 10px">

.. app_card::
   :title: Hello world
   :description: Something supereererererererererer long asdf asf asdfasdf a 
   :width: 280
   :image: https://media3.giphy.com/media/KgEzIaqjorVRVGSvpU/giphy.gif
   :preview: https://lightning.ai
   :deploy: https://lightning.ai
   :target: https://apple.com

.. app_card::
   :title: Train a model
   :width: 280
   :image: https://media3.giphy.com/media/KgEzIaqjorVRVGSvpU/giphy.gif
   :preview: https://lightning.ai
   :deploy: https://lightning.ai
   :target: https://apple.com

.. app_card::
   :title: Deploy a model
   :width: 280
   :image: https://media3.giphy.com/media/KgEzIaqjorVRVGSvpU/giphy.gif
   :preview: https://lightning.ai
   :deploy: https://lightning.ai
   :target: https://apple.com

.. raw:: html

      </div>
   </div>

----

************************
Build modular components
************************
Build modular, self-contained Lightning components that can do anything from 
train models, and deploy models to run web user interfaces. A Lightning component 
manages its own infrastructure, cloud costs, networking and more. Components
can be connected to form a full stack AI app that we call a Lightning App.

Image:
[Component] -> many components working together -> end product

.. lit_tabs::
   :code_files: code_a.py; 
   :highlights: 6, 11
   :app_id: abc123
   :height: 385px

|

.. raw:: html

   <p>Get inspired by community-built Lightning components on our <a href='https://lightning.ai/components' target="_blank">ecosystem</a></p>

----

********************
Get started in steps
********************
Learn to go from Lightning components to full stack AI apps step-by-step.

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Level 1: Build a Lightning component
   :description: Build a Lightning component that wraps Python code and runs on the cloud.
   :col_css: col-md-12
   :button_link: levels/basic/build_a_lightning_component.html
   :height: 160
   :tag: All users

.. raw:: html

        </div>
    </div>
----

*****************
Install Lightning
*****************
Use Python 3.8.x or later. We also recommend you install in a virtual environment (`learn how <install_beginner.rst>`_).

.. code:: bash

   python -m pip install -U lightning


For Mac M1/M2/M3, windows or custom installs, read the :ref:`advanced install <install>` guide.

----

***********************
Current Lightning Users
***********************

.. raw:: html

   <div class="display-card-container" style="padding: 0 10px 0 10px">
      <div class="row">

.. displayitem::
   :header: Level-up your skills
   :description: From Basics to Advanced Skills
   :col_css: col-md-6
   :button_link: levels/basic/index.html
   :height: 180

.. displayitem::
   :header: API Reference
   :description: Detailed description of each API package
   :col_css: col-md-6
   :button_link: api_references.html
   :height: 180

.. displayitem::
   :header: Glossary
   :description: Discover Lightning App Concepts
   :col_css: col-md-6
   :button_link: glossary/index.html
   :height: 180

.. displayitem::
   :header: Start from Ready-to-Run Template Apps
   :description: Jump-start your project's development
   :col_css: col-md-6
   :button_link: get_started/jumpstart_from_app_gallery.html
   :height: 180

.. displayitem::
   :header: Add Component made by others to your App
   :description: Add more functionalities to your projects
   :col_css: col-md-6
   :button_link: get_started/jumpstart_from_component_gallery.html
   :height: 180

.. displayitem::
   :header: Hands-on Examples
   :description: Learn by building Apps and Components.
   :col_css: col-md-6
   :button_link: examples/hands_on_example.html
   :height: 180

.. displayitem::
   :header: Common Workflows
   :description: Learn how to do ...
   :col_css: col-md-6
   :button_link: workflows/index.html
   :height: 180

.. raw:: html

   <hr class="docutils" style="margin: 50px 0 50px 0">

.. raw:: html

   <div style="display:none">

.. toctree::
   :maxdepth: 1
   :caption: Home

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

   Examples <examples/index>
   How to... <workflows/index>
   Glossary <glossary/index>
