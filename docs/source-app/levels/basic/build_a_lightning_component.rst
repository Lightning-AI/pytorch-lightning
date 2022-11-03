##########################################
Package your code in a lightning component
##########################################

**Prereqs:** You know *basic* Python.

**Goal:** In this guide you'll learn to develop `a Lightning component <https://lightning.ai/components>`_.

.. join_slack::
   :align: left
----

*********************************
Why you need Lightning components
*********************************
ML workflows and full stack AI apps require many pieces working such as training, deploying, data annotation. However, this tight coupling
can lead to monoliths that are hard to scale or many microservices that are hard to monitor, coordinate
and scale. A Lightning component is a self-contained piece of code (ie: a microservice) that executes your code
and manages its own infrastructure, auto-scaling and costs for you.

.. raw:: html

   <div style="display: flex; align-items: center; justify-content: center;">
      <img src="https://lightning-ai-docs.s3.amazonaws.com/intro_components.gif" style="max-width: 800px"></img>
   </div>

|

By using Lightning components you can focus on application logic without dealing with cloud operations
or even how components speak to each other. Together, Lightning components form a Lightning App that
gives you these benefits:

.. collapse:: Build systems not scripts

   |

   The Lightning structure forces best practices so you don't have to be an expert production engineer.
   Although it feels like you're writing a script, you are actually building a production-ready system.


.. collapse:: Cost control

   |

   The component run-time has been optimized for cost management to support the largest machine-learning workloads.
   Lower your cloud bill with machines that shut down or spin up faster.

.. collapse:: For beginners: Code like an expert

   |

   Lightning embeds the best practices of building production-ready full stack AI apps into your
   coding experience. You can write code like you normally do, and the Lightning structure
   ensures your code is implicitely production ready... even if you're just doing research.


.. collapse:: For experts: Scale with full control

   |

   if you know what you are doing, Lightning gives you full control to manage your own
   scaling logic, fault-tolerance and even pre-provisioning, all from Python. We even give you
   full flexibility to use tools like `terraform <../../cloud/customize_a_lightning_cluster.html>`_ to optimize cloud clusters for your Lightning apps.

.. collapse:: Packaged code

   |

   Lightning apps bundles components into an app that runs in any environment. The same code will run on your laptop,
   or any cloud or private clusters. You don't have to think about the cluster or know anything about the cloud.

.. collapse:: Rapid iteration

   |

   Iterate through ideas in hours not months because you don't have to learn a million other concepts that the components
   handle for you such as kubernetes, cost management, auto-scaling and more.

.. collapse:: Modularity

   |

   Components are modular and inter-operable by design. Leverage our vibrant community of components so you don't
   have to build each piece of the system yourself.

----

*****************
Install Lightning
*****************
First, install Lightning.

.. code:: bash

    python -m pip install -U lightning

.. collapse:: Mac M1/M2/M3 and Windows users

   |

   **Mac**

   To install on Mac, set these 2 environment variables

   .. code-block:: bash

      # needed for M1/M2/M3
      export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
      export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

      python -m pip install -U lightning

   **Windows users**

   To install on Windows:

   - setup an alias for Python: python=python3
   - Add the root folder of Lightning to the Environment Variables to PATH

----

**************************
Build your first component
**************************
A Lightning component organizes Python code so it can run on the cloud and be connected with other components to form a Lightning App.
Pick one of these components to run:

.. lit_tabs::
   :titles: Hello CPU world; Hello GPU (accelerated) world; Train PyTorch on a cloud GPU; Train PyTorch ⚡ on cloud GPUs; Deploy a model on cloud GPUs; Run a model script; Build a model web UI
   :code_files: ./hello_components/hello_world.py; ./hello_components/hello_world_gpu.py; ./hello_components/train_pytorch.py; ./hello_components/train_ptl.py; ./hello_components/deploy_model.py; ./hello_components/run_script.py; ./hello_components/build_demo.py
   :highlights: 7; 10, 11; 3, 7-23;3;4;5;6
   :app_id: abc123
   :tab_rows: 4
   :height: 550px

|

Components run the same on the cloud and locally on your choice of hardware.

.. lit_tabs::
   :titles: Lightning Cloud (fully-managed); Your AWS account; Your own hardware
   :code_files: ./hello_components/code_run_cloud.bash; ./hello_components/code_run_cloud_yours.bash; ./hello_components/code_run_local.bash
   :tab_rows: 4
   :height: 195px

----

************
Key features
************
You now know enough to build a self-contained component that runs any Python code on the cloud that can be connected to form a
powerful Lightning app. Here are a few key features available to super-charge your work:

.. lit_tabs::
   :titles: 15+ accelerators; Auto-stop idle machines; Auto-timeout submitted work; Use spot machines (~70% discount); Work with massive datasets; Mount cloud storage; Use a custom container
   :code_files: ./key_features/accelerators.py; ./key_features/idle_machine.py; ./key_features/auto_timeout.py; ./key_features/spot.py; ./key_features/massive_dataset.py; ./key_features/mount_data.py; ./key_features/custom_container.py;
   :highlights: 10;10;10;10;10;2,6,9, 10; 7
   :app_id: abc123
   :tab_rows: 3
   :height: 430px

----

***************************
Use the community ecosystem
***************************
Lightning has a vibrant collection of community-built components you can use as templates or to inspire you.


----

************************
Next: Connect components
************************
Now you can build components. To build powerful full stack AI apps you'll need to learn to connect them together.

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Level 2: Connect components
   :description: Learn to connect components
   :col_css: col-md-12
   :button_link: connect_lightning_components.html
   :height: 150
   :tag: beginner

.. raw:: html

        </div>
    </div>
