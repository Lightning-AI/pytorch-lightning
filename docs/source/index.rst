.. lightning documentation master file, created by
   sphinx-quickstart on Sat Sep 19 16:37:02 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

############################
Welcome to ⚡ Lightning Apps
############################

.. twocolumns::
   :left:
      .. image:: https://pl-flash-data.s3.amazonaws.com/assets_lightning/Lightning.gif
         :alt: Animation showing how to convert a standard training loop to a Lightning loop
   :right:
      The `open-source Lightning framework <https://github.com/Lightning-AI/lightning>`_ gives ML Researchers and Data Scientists, the fastest & most flexible
      way to iterate on ML research ideas and deliver scalable ML systems with the performance enterprises requires at the same time.

.. join_slack::
   :align: center
   :margin: 0

----

*****************
Install Lightning
*****************

.. code-block:: bash

   pip install lightning


Or read the :ref:`advanced install <install>` guide.

----

***********
Get Started
***********

.. raw:: html

   <br />
   <div class="display-card-container">
      <div class="row">

.. displayitem::
   :header: Discover what Lightning Apps can do in 5 min
   :description: Browse through mind-blowing ML Systems
   :col_css: col-md-6
   :button_link: source-app/get_started/what_app_can_do.html
   :height: 180

.. displayitem::
   :header: Build and Train a Model
   :description: Discover PyTorch Lightning and train your first Model.
   :col_css: col-md-6
   :button_link: source-app/get_started/build_model.html
   :height: 180

.. displayitem::
   :header: Evolve a Model into an ML System
   :description: Develop an App to train a model in the cloud
   :col_css: col-md-6
   :button_link: source-app/get_started/training_with_apps.html
   :height: 180

.. displayitem::
   :header: Start from an ML system template
   :description: Learn about Apps, from a template.
   :col_css: col-md-6
   :button_link: source-app/get_started/go_beyond_training.html
   :height: 180

.. raw:: html

      </div>
   </div>

----

***********************
Current Lightning Users
***********************

.. raw:: html

   <br />

Build with Template(s) from the App & Component Gallery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Start from Ready-to-Run Template Apps
   :description: Jump-start your project's development
   :col_css: col-md-6
   :button_link: source-app/get_started/jumpstart_from_app_gallery.html
   :height: 180

.. displayitem::
   :header: Add Component made by others to your App
   :description: Add more functionalities to your projects
   :col_css: col-md-6
   :button_link: source-app/get_started/jumpstart_from_component_gallery.html
   :height: 180

.. raw:: html

      </div>
   </div>
   <br />

Keep Learning
^^^^^^^^^^^^^

.. raw:: html

   <div class="display-card-container">
      <div class="row">

.. displayitem::
   :header: Level-up with PyTorch Lightning
   :description: PyTorch Lightning Tutorials
   :col_css: col-md-6
   :button_link: https://pytorch-lightning.readthedocs.io/en/latest/expertise_levels.html
   :height: 180

.. displayitem::
   :header: Level-up with Lightning Apps
   :description: From Basics to Advanced Skills
   :col_css: col-md-6
   :button_link: source-app/levels/basic/index.html
   :height: 180

.. displayitem::
   :header: API Reference
   :description: Detailed description of each API package
   :col_css: col-md-6
   :button_link: source-app/api_references.html
   :height: 180

.. displayitem::
   :header: Hands-on Examples
   :description: Learn by building Apps and Components.
   :col_css: col-md-6
   :button_link: source-app/examples/hands_on_example.html
   :height: 180

.. displayitem::
   :header: Common Workflows
   :description: Learn how to do ...
   :col_css: col-md-6
   :button_link: source-app/workflows/index.html
   :height: 180

.. displayitem::
   :header: Glossary
   :description: Discover Lightning App Concepts
   :col_css: col-md-6
   :button_link: source-app/glossary/index.html
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
   :caption: Get Started

   source-app/installation
   source-app/get_started/lightning_apps_intro

.. toctree::
   :maxdepth: 1
   :caption: App Building Skills

   Basic <source-app/levels/basic/index>
   Intermediate <source-app/levels/intermediate/index>
   Advanced <source-app/levels/advanced/index>

.. toctree::
   :maxdepth: 1
   :caption: Examples

   Develop a DAG <source-app/examples/dag/dag>
   Develop a File Server <source-app/examples/file_server/file_server>
   Develop a Github Repo Script Runner <source-app/examples/github_repo_runner/github_repo_runner>
   Develop a HPO Sweeper <source-app/examples/hpo/hpo>
   Develop a Model Server <source-app/examples/model_server_app/model_server_app>

..
   [Docs under construction] Build a data exploring app  <examples/data_explore_app>
   [Docs under construction] Build a ETL app  <examples/etl_app>
   [Docs under construction] Build a model deployment app  <examples/model_deploy_app>
   [Docs under construction] Build a research demo app  <examples/research_demo_app>

.. toctree::
   :maxdepth: 1
   :caption: How to...

   Add a web user interface (UI) <source-app/workflows/add_web_ui/index>
   Add a web link  <source-app/workflows/add_web_link>
   Arrange app tabs <source-app/workflows/arrange_tabs/index>
   Develop a Lightning App <source-app/workflows/build_lightning_app/index>
   Develop a Lightning Component <source-app/workflows/build_lightning_component/index>
   Cache Work run calls  <source-app/workflows/run_work_once>
   Customize your cloud compute <source-app/core_api/lightning_work/compute>
   Extend an existing app <source-app/workflows/extend_app>
   Publish a Lightning component <source-app/workflows/build_lightning_component/publish_a_component>
   Run a server within a Lightning App <source-app/workflows/add_server/index>
   Run an App on the cloud <source-app/workflows/run_app_on_cloud/index>
   Run work in parallel <source-app/workflows/run_work_in_parallel>
   Share an app  <source-app/workflows/share_app>
   Share files between components <source-app/workflows/share_files_between_components>

..
   [Docs under construction] Add a Lightning component  <workflows/add_components/index>
   [Docs under construction] Debug a distributed cloud app locally <workflows/debug_locally>
   [Docs under construction] Enable fault tolerance  <workflows/enable_fault_tolerance>
   [Docs under construction] Run components on different hardware  <workflows/run_components_on_different_hardware>
   [Docs under construction] Schedule app runs  <workflows/schedule_apps>
   [Docs under construction] Test an app  <workflows/test_an_app>

.. toctree::
   :maxdepth: 1
   :caption: Core API Reference

   LightningApp <source-app/core_api/lightning_app/index>
   LightningFlow <source-app/core_api/lightning_flow>
   LightningWork <source-app/core_api/lightning_work/index>

.. toctree::
   :maxdepth: 1
   :caption: Addons API Reference

   source-app/api_reference/components
   source-app/api_reference/frontend
   source-app/api_reference/runners
   source-app/api_reference/storage

.. toctree::
   :maxdepth: 1
   :caption: Glossary

   App Components Tree <source-app/glossary/app_tree>
   Build Configuration <source-app/glossary/build_config/build_config>
   DAG <source-app/glossary/dag>
   Event Loop <source-app/glossary/event_loop>
   Environment Variables <source-app/glossary/environment_variables>
   Frontend <source-app/workflows/add_web_ui/glossary_front_end.rst>
   Sharing Components <source-app/glossary/sharing_components>
   Scheduling <source-app/glossary/scheduling.rst>
   Storage <source-app/glossary/storage/storage.rst>
   UI <source-app/workflows/add_web_ui/glossary_ui.rst>

..
   [Docs under construction] Debug an app <glossary/debug_app>
   [Docs under construction] Distributed front-ends <glossary/distributed_fe>
   [Docs under construction] Distributed hardware <glossary/distributed_hardware>
   [Docs under construction] Fault tolerance <glossary/fault_tolerance>
