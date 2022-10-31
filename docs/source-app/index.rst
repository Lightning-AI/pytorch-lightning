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

   python -m pip install -U lightning


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
   :button_link: get_started/what_app_can_do.html
   :height: 180

.. displayitem::
   :header: Build and Train a Model
   :description: Discover PyTorch Lightning and train your first Model.
   :col_css: col-md-6
   :button_link: get_started/build_model.html
   :height: 180

.. displayitem::
   :header: Evolve a Model into an ML System
   :description: Develop an App to train a model in the cloud
   :col_css: col-md-6
   :button_link: get_started/training_with_apps.html
   :height: 180

.. displayitem::
   :header: Start from an ML system template
   :description: Learn about Apps, from a template.
   :col_css: col-md-6
   :button_link: get_started/go_beyond_training.html
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
   :button_link: get_started/jumpstart_from_app_gallery.html
   :height: 180

.. displayitem::
   :header: Add Component made by others to your App
   :description: Add more functionalities to your projects
   :col_css: col-md-6
   :button_link: get_started/jumpstart_from_component_gallery.html
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
   :button_link: levels/basic/index.html
   :height: 180

.. displayitem::
   :header: API Reference
   :description: Detailed description of each API package
   :col_css: col-md-6
   :button_link: api_references.html
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

.. displayitem::
   :header: Glossary
   :description: Discover Lightning App Concepts
   :col_css: col-md-6
   :button_link: glossary/index.html
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

   installation
   get_started/lightning_apps_intro

.. toctree::
   :maxdepth: 1
   :caption: App Building Skills

   Basic <levels/basic/index>
   Intermediate <levels/intermediate/index>
   Advanced <levels/advanced/index>

.. toctree::
   :maxdepth: 1
   :caption: Examples

   Develop a DAG <examples/dag/dag>
   Develop a File Server <examples/file_server/file_server>
   Develop a Github Repo Script Runner <examples/github_repo_runner/github_repo_runner>
   Develop a Model Server <examples/model_server_app/model_server_app>

..
   [Docs under construction] Build a data exploring app  <examples/data_explore_app>
   [Docs under construction] Build a ETL app  <examples/etl_app>
   [Docs under construction] Build a model deployment app  <examples/model_deploy_app>
   [Docs under construction] Build a research demo app  <examples/research_demo_app>

.. toctree::
   :maxdepth: 1
   :caption: How to...

   Access the App State <workflows/access_app_state/access_app_state>
   Add a web user interface (UI) <workflows/add_web_ui/index>
   Add a web link  <workflows/add_web_link>
   Add encrypted secrets <glossary/secrets>
   Arrange app tabs <workflows/arrange_tabs/index>
   Develop a Command Line Interface (CLI) <workflows/build_command_line_interface/index>
   Develop a Lightning App <workflows/build_lightning_app/index>
   Develop a Lightning Component <workflows/build_lightning_component/index>
   Develop a REST API <workflows/build_rest_api/index>
   Cache Work run calls  <workflows/run_work_once>
   Customize your cloud compute <core_api/lightning_work/compute>
   Extend an existing app <workflows/extend_app>
   Publish a Lightning component <workflows/build_lightning_component/publish_a_component>
   Run a server within a Lightning App <workflows/add_server/index>
   Run an App on the cloud <workflows/run_app_on_cloud/index>
   Run Apps on your cloud account (BYOC) <workflows/byoc/index>
   Run work in parallel <workflows/run_work_in_parallel>
   Save files <glossary/storage/drive.rst>
   Share an app  <workflows/share_app>
   Share files between components <workflows/share_files_between_components>

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
   :caption: Glossary

   App Components Tree <glossary/app_tree>
   Build Configuration <glossary/build_config/build_config>
   Command Line Interface (CLI) <glossary/command_lines/command_lines>
   DAG <glossary/dag>
   Event Loop <glossary/event_loop>
   Environment Variables <glossary/environment_variables>
   Encrypted Secrets <glossary/secrets>
   Frontend <workflows/add_web_ui/glossary_front_end.rst>
   Apple and Android mobile devices with Lighting Apps <glossary/ios_and_android>
   REST API <glossary/restful_api/restful_api>
   Sharing Components <glossary/sharing_components>
   Scheduling <glossary/scheduling.rst>
   Storage <glossary/storage/storage.rst>
   UI <workflows/add_web_ui/glossary_ui.rst>

..
   [Docs under construction] Debug an app <glossary/debug_app>
   [Docs under construction] Distributed front-ends <glossary/distributed_fe>
   [Docs under construction] Distributed hardware <glossary/distributed_hardware>
   [Docs under construction] Fault tolerance <glossary/fault_tolerance>
