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
   Develop a Lightning App <workflows/build_lightning_app/index>
   Develop a Lightning Component <workflows/build_lightning_component/index>
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
   DAG <glossary/dag>
   Event Loop <glossary/event_loop>
   Environment Variables <glossary/environment_variables>
   Encrypted Secrets <glossary/secrets>
   Frontend <workflows/add_web_ui/glossary_front_end.rst>
   Sharing Components <glossary/sharing_components>
   Scheduling <glossary/scheduling.rst>
   Storage <glossary/storage/storage.rst>
   UI <workflows/add_web_ui/glossary_ui.rst>

..
   [Docs under construction] Debug an app <glossary/debug_app>
   [Docs under construction] Distributed front-ends <glossary/distributed_fe>
   [Docs under construction] Distributed hardware <glossary/distributed_hardware>
   [Docs under construction] Fault tolerance <glossary/fault_tolerance>

.. toctree::
   :maxdepth: 1
   :name: start
   :caption: Get Started

   starter/introduction
   starter/installation


.. toctree::
   :maxdepth: 2
   :name: levels
   :caption: Level Up

   levels/core_skills
   levels/intermediate
   levels/advanced
   levels/expert

.. toctree::
   :maxdepth: 2
   :name: pl_docs
   :caption: Core API

   common/lightning_module
   common/trainer

.. toctree::
   :maxdepth: 2
   :name: api
   :caption: API Reference

   api_references

.. toctree::
   :maxdepth: 1
   :name: Common Workflows
   :caption: Common Workflows

   Avoid overfitting <common/evaluation>
   model/build_model.rst
   common/hyperparameters
   common/progress_bar
   deploy/production
   advanced/training_tricks
   cli/lightning_cli
   tuning/profiler
   Manage experiments <visualize/logging_intermediate>
   Organize existing PyTorch into Lightning <starter/converting>
   clouds/cluster
   Save and load model progress <common/checkpointing>
   Save memory with half-precision <common/precision>
   Training over the internet <strategies/hivemind>
   advanced/model_parallel
   clouds/cloud_training
   Train on single or multiple GPUs <accelerators/gpu>
   Train on single or multiple HPUs <accelerators/hpu>
   Train on single or multiple IPUs <accelerators/ipu>
   Train on single or multiple TPUs <accelerators/tpu>
   Train on MPS <accelerators/mps>
   Use a pretrained model <advanced/pretrained>
   model/own_your_loop

.. toctree::
   :maxdepth: 1
   :name: Glossary
   :caption: Glossary

   Accelerators <extensions/accelerator>
   Callback <extensions/callbacks>
   Checkpointing <common/checkpointing>
   Cluster <clouds/cluster>
   Cloud checkpoint <common/checkpointing_advanced>
   Console Logging <common/console_logs>
   Debugging <debug/debugging>
   Early stopping <common/early_stopping>
   Experiment manager (Logger) <visualize/experiment_managers>
   Fault tolerant training  <clouds/fault_tolerant_training>
   Finetuning <advanced/finetuning>
   Flash <https://lightning-flash.readthedocs.io/en/stable/>
   Grid AI <clouds/cloud_training>
   GPU <accelerators/gpu>
   Half precision <common/precision>
   HPU <accelerators/hpu>
   Inference <deploy/production_intermediate>
   IPU <accelerators/ipu>
   Lightning CLI <cli/lightning_cli>
   Lightning Lite <model/build_model_expert>
   LightningDataModule <data/datamodule>
   LightningModule <common/lightning_module>
   Lightning Transformers <https://pytorch-lightning.readthedocs.io/en/stable/ecosystem/transformers.html>
   Log <visualize/loggers>
   Loops <extensions/loops>
   TPU <accelerators/tpu>
   Metrics <https://torchmetrics.readthedocs.io/en/stable/>
   Model <model/build_model.rst>
   Model Parallel <advanced/model_parallel>
   Collaborative Training <strategies/hivemind>
   Plugins <extensions/plugins>
   Progress bar <common/progress_bar>
   Production <deploy/production_advanced>
   Predict <deploy/production_basic>
   Pretrained models <advanced/pretrained>
   Profiler <tuning/profiler>
   Pruning and Quantization <advanced/pruning_quantization>
   Remote filesystem and FSSPEC <common/remote_fs>
   Strategy <extensions/strategy>
   Strategy registry <advanced/strategy_registry>
   Style guide <starter/style_guide>
   Sweep <clouds/run_intermediate>
   SWA <advanced/training_tricks>
   SLURM <clouds/cluster_advanced>
   Transfer learning <advanced/transfer_learning>
   Trainer <common/trainer>
   Torch distributed <clouds/cluster_intermediate_2>

.. toctree::
   :maxdepth: 1
   :name: Hands-on Examples
   :caption: Hands-on Examples
   :glob:

   PyTorch Lightning 101 class <https://www.youtube.com/playlist?list=PLaMu-SDt_RB5NUm67hU2pdE75j6KaIOv2>
   From PyTorch to PyTorch Lightning [Blog] <https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09>
   From PyTorch to PyTorch Lightning [Video] <https://www.youtube.com/watch?v=QHww1JH7IDU>

.. toctree::
   :maxdepth: 1
   :name: Community
   :caption: Community

   generated/CODE_OF_CONDUCT.md
   generated/CONTRIBUTING.md
   generated/BECOMING_A_CORE_CONTRIBUTOR.md
   governance
