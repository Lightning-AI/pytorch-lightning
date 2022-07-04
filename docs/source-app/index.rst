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
      Lightning is a distributed, modular, free, and open framework for
      building all AI applications where the components you want, interact together.
      The concept of Lightning apps was designed to help you focus on the what you care about,
      and automate the rest.
      The Lightning framework can be used for any type of AI app, from a simple demo to a production pipeline.

.. raw:: html

   <div class="row" style='font-size: 14px'>
      <div class='col-md-6'>
      </div>
      <div class='col-md-6'>

.. join_slack::
   :align: center
   :margin: 0

.. raw:: html

      </div>
   </div>

----

******************
Install Lightning
******************


.. code-block:: bash

   pip install lightning

----

******************
Get Started
******************

.. raw:: html

    <div class="tutorials-callout-container">
        <div class="row">

.. Add callout items below this line

.. customcalloutitem::
   :header: Build a Lightning App in 15 minutes
   :description: Learn the 4 key steps to build a Lightning app.
   :button_link:  lightning_apps_intro.html

.. raw:: html

        </div>
    </div>

.. End of callout item section

.. raw:: html

   <hr class="docutils" style="margin: 50px 0 50px 0">

.. raw:: html

   <div style="display:none">

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   read_me_first
   installation
   lightning_apps_intro

.. toctree::
   :maxdepth: 1
   :caption: App Building Skills

   Basic <levels/basic/index>
   Intermediate <levels/intermediate/index>
   Advanced <levels/advanced/index>

.. toctree::
   :maxdepth: 1
   :caption: Hands-on Examples

   Build a DAG <examples/dag/dag>
   Build a File Server <examples/file_server/file_server>
   Build a Github Repo Script Runner <examples/github_repo_runner/github_repo_runner>
   Build a HPO Sweeper <examples/hpo/hpo>
   Build a Model Server <examples/model_server_app/model_server_app>

..
   [Docs under construction] Build a data exploring app  <examples/data_explore_app>
   [Docs under construction] Build a ETL app  <examples/etl_app>
   [Docs under construction] Build a model deployment app  <examples/model_deploy_app>
   [Docs under construction] Build a research demo app  <examples/research_demo_app>

.. toctree::
   :maxdepth: 1
   :caption: App development workflows

   Access the app state </workflows/access_app_state/access_app_state.rst>
   Add a web user interface (UI) <workflows/add_web_ui/index>
   Add a web link  <workflows/add_web_link>
   Arrange app tabs <workflows/arrange_tabs/index>
   Build a Lightning app <workflows/build_lightning_app/index>
   Build a Lightning component <workflows/build_lightning_component/index>
   Cache Work run calls  <workflows/run_work_once>
   Extend an existing app <workflows/extend_app>
   Publish a Lightning component <workflows/build_lightning_component/publish_a_component>
   Run a server within a Lightning App <workflows/add_server/index>
   Run an app on the cloud <workflows/run_app_on_cloud/index>
   Run work in parallel <workflows/run_work_in_parallel>
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
   DAG <glossary/dag>
   Event loop <glossary/event_loop>
   Environment Variables <glossary/environment_variables>
   Front-end <workflows/add_web_ui/glossary_front_end.rst>
   Sharing Components <glossary/sharing_components>
   Scheduling <glossary/scheduling.rst>
   Storage <glossary/storage/storage.rst>
   UI <workflows/add_web_ui/glossary_ui.rst>

..
   [Docs under construction] Debug an app <glossary/debug_app>
   [Docs under construction] Distributed front-ends <glossary/distributed_fe>
   [Docs under construction] Distributed hardware <glossary/distributed_hardware>
   [Docs under construction] Fault tolerance <glossary/fault_tolerance>
