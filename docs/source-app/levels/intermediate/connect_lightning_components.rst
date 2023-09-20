####################################################
Level 4: Connect components into a full stack AI app
####################################################

**Audience:** Users who want to build apps with multiple components.

**Prereqs:** You know how to :doc:`build a component <../basic/build_a_lightning_component>`.

----

****************************
What is a full stack AI app?
****************************
In the ML world, workflows coordinate multiple pieces of code working together. In Lightning,
when we coordinate 2 or more :doc:`Lightning components <../basic/build_a_lightning_component>` working together,
we instead call it a Lightning App. The difference will become more obvious when we introduce reactive
workflows in the advanced section.

For the time being, we'll go over how to coordinate 2 components together in a traditional workflow setting
and explain how it works.

.. note:: If you've used workflow tools for Python, this page describes conventional DAGs.
        In :doc:`level 6 <run_lightning_work_in_parallel>`, we introduce reactive workflows that generalize beyond DAGs
        so you can build complex systems without much effort.

----

***********
The toy app
***********

In this app, we define two components that run across 2 separate machines. One to train a model on a GPU machine and one to analyze the model
on a separate CPU machine. We save money by stopping the GPU machine when the work is done.

.. lit_tabs::
   :titles: Import Lightning; Define Component 1;  Define Component 2; Orchestrator; Connect component 1; Connect component 2; Implement run; Train; Analyze; Define app placeholder
   :descriptions: First, import Lightning; This component trains a model on a GPU machine; This component analyzes a model on a CPU machine; Define the LightningFlow that orchestrates components; Component 1 will run on a CPU machine; Component 2 will run on an accelerated GPU machine; Describe the workflow in the run method; Training runs first and completes; Analyze runs after training completes; This allows the app to be runnable
   :code_files: ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py
   :highlights: 2; 5-7; 9-11; 13; 16; 17; 19; 20; 21; 23
   :enable_run: true
   :tab_rows: 4
   :height: 460px

|

Now run the app:

.. lit_tabs::
   :titles: Run on Lightning cloud; Your own hardware
   :descriptions: Run to see these 2 components execute on separate machines 🤯; Run it locally without code changes 🤯🤯;
   :code_files: ./level_2_scripts/code_run_cloud.bash; ./level_2_scripts/code_run_local.bash
   :tab_rows: 7
   :height: 195px

|

Now you can develop distributed cloud apps on your laptop 🤯🤯🤯🤯!

----

*************
Now you know:
*************

Without going out of your way, you're now doing the following: (Hint: Click **visualize** to see an animation describing the code).

.. lit_tabs::
   :titles: Orchestration; Distributed cloud computing; Multi-machine communication; Multi-machine communication; Multi-cloud;
   :descriptions: Define orchestration in Python with full control-flow; The two pieces of independent Python code ran on separate machines 🤯🤯; The text "CPU machine 1" was sent from the flow machine to the machine running the TrainComponent;  The text "GPU machine 2" was sent from the flow machine to the machine running the AnalyzeComponent; The full Lightning app can move across clusters and clouds
   :code_files: ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py;
   :tab_rows: 4
   :highlights: 19-21; 16-17; 20; 21
   :images: <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/orchestration.gif" style="max-height: 430px; width: auto"></img> | <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/distributed_computing.gif" style="max-height: 430px; width: auto"></img> | <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/multi_machine_comms.gif" style="max-height: 430px; width: auto"></img> | <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/multi_machine_comms.gif" style="max-height: 430px; width: auto"></img> | <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/multi_cloud.gif" style="max-height: 430px; width: auto"></img>
   :height: 470px

----

*********************
Maintain full control
*********************
Although we've abstracted the infrastructure, you still have full control when you need it:

.. lit_tabs::
   :titles: Scheduler; Crontab syntax; Auto-scaling; Organized Python; Full terraform control;
   :descriptions: Although you can use Python timers, we have a scheduler short-hand; You can also use full cron syntax; Code your own auto-scaling syntax (Lightning plays well with Kubernetes); *Remember* components organize ANY Python code which can even call external non-python scripts such as optimized C++ model servers ;Experts have the option to use terraform to configure Lightning clusters
   :code_files: ./level_2_scripts/hello_app_scheduler.py; ./level_2_scripts/hello_app_cron.py; ./level_2_scripts/hello_app_auto_scale.py; ./level_2_scripts/organized_app_python.py;
   :tab_rows: 4
   :highlights: 24; 24; 21, 24, 27, 28; 9, 16, 17
   :height: 700px

----

*************************
Next: Review how to debug
*************************
The next levels does a 2 minute review to make sure you know how to debug a Lightning app.

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Level 5: Debug a Lightning App
   :description: Learn to debug a lightning app.
   :button_link: debug_a_lightning_app.html
   :col_css: col-md-12
   :height: 170
   :tag: 10 minutes

.. raw:: html

        </div>
    </div>
