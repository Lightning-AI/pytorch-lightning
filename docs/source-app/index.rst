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

*******************************************************
Build full stack AI apps from modular Python components
*******************************************************

.. raw:: html

   <div class="row">
      <div class='col-md-5' style="padding-top: 40px">
      <p>
         Lightning makes it painless to build full stack AI apps for the cloud. Implement Lightning components that encapsulate 
         each part of your application to make it modular, interoperable and maintainable. 
         Each Lightning component manages its own infrastructure, cloud costs, networking and more, so you can focus on the application logic.
      </p>
      </div>
      <div class='col-md-7'>

.. raw:: html

   <video style="height:auto; max-width: 100%; max-height: 450px" controls autoplay muted playsinline src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/hello_apps.m4v"></video>

.. raw:: html

      </div>
   </div>

********************************
Define your Lightning Components
********************************
Each Lightning component manages its own infrastructure, cloud costs, networking and more, so you can focus on the application logic.

.. lit_tabs::
   :titles: ; Example; Example; Example;
   :descriptions: Add your code to a Lightning component; Train models on cloud GPUs; Serve models; Run R, Python, C++, Matlab scripts
   :code_files: code_a.py; code_a_train_models.py; code_a_deploy_models.py; code_a_subprocess.py
   :highlights: 6, 11; 7, 11; 6-8; 7, 8
   :images: <img src='https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/cloud_run_all.gif' style="width: auto; height: 95%; padding: 10px"> | <img src='https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/cloud_run_all.gif' style="width: 450px; height: auto"> | <img src='https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/cloud_run_all.gif' style="width: 450px; height: auto"> | <img src='https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/cloud_run_all.gif' style="width: 450px; height: auto">
   :side_image_width: 58%
   :app_id: abc123
   :tab_rows: 4
   :height: 385px

----

******************************************
Compose components into full stack AI apps
******************************************
Components "connect" into full stack AI apps which we call Lightning apps. 
Lightning apps can be as simple as sequential ML workflows without GUIs, or as complex as
a generative AI app managing its own full react UI with models deploying in real-time.

.. lit_tabs::
   :titles: ; Example; Example; Example;
   :descriptions: Compose modules into an app; Traditional sequential workflow; Human in the loop; Example: Something else 
   :code_files: code_a.py; code_a_train_models.py; code_a_deploy_models.py; code_a_subprocess.py
   :highlights: 6, 11; 7, 11; 6-8; 7, 8
   :tab_rows: 4
   :height: 315px

----

*********************************
Run on the cloud or your hardware
*********************************
Lightning apps are portable! take them to the cloud or cluster of your choice.

.. lit_tabs::
   :descriptions: Lightning Cloud (fully-managed); Your AWS account; Your own hardware
   :code_files: code_run_cloud.bash; code_run_cloud_yours.bash; code_run_local.bash
   :tab_rows: 4
   :height: 195px

----

***********
Get Started
***********
Pick a tutorial to build your first Lightning App.

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

.. app_card::
   :title: Build a full stack AI app
   :width: 280
   :image: https://media3.giphy.com/media/KgEzIaqjorVRVGSvpU/giphy.gif
   :preview: https://lightning.ai
   :deploy: https://lightning.ai
   :target: https://apple.com

.. app_card::
   :title: Build a cloud ML platform
   :width: 280
   :image: https://media3.giphy.com/media/KgEzIaqjorVRVGSvpU/giphy.gif
   :preview: https://lightning.ai
   :deploy: https://lightning.ai
   :target: https://apple.com

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
   :caption: Get Started

   installation
   Lightning in 15 minutes <levels/basic/lightning_in_15_minutes>

.. toctree::
   :maxdepth: 1
   :caption: Learn Lightning

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

   Android Lightning App <glossary/ios_and_android>
   App Components Tree <glossary/app_tree>
   Build Configuration <glossary/build_config/build_config>
   Command Line Interface (CLI) <glossary/command_lines/command_lines>
   DAG <glossary/dag>
   Event Loop <glossary/event_loop>
   Environment Variables <glossary/environment_variables>
   Encrypted Secrets <glossary/secrets>
   Frontend <workflows/add_web_ui/glossary_front_end.rst>
   iOS Lightning App <glossary/ios_and_android>
   Lightning App <core_api/lightning_app/index.rst>
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
