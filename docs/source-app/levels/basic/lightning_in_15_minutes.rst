#######################
Lightning in 15 minutes
#######################
**Prereqs:** You know *basic* Python.

**Goal:** In this guide you'll learn to develop `full stack AI apps <https://lightning.ai/>`_ with Lightning.

.. join_slack::
   :align: left
----

******************
What is Lightning?
******************
Lightning is the `open-source framework <https://github.com/Lightning-AI/lightning>`_ that lets you run Python code `on the cloud <https://lightning.ai/>`_ to build **full stack AI apps** like these:

.. raw:: html

   <div class="w3-bar w3-black">
      <button class="w3-bar-item w3-button" onclick="openCity('London')">London</button>
      <button class="w3-bar-item w3-button" onclick="openCity('Paris')">Paris</button>
      <button class="w3-bar-item w3-button" onclick="openCity('Tokyo')">Tokyo</button>
   </div>
   <div id="London" class="city">
   <h2>London</h2>
   <p>London is the capital of England.</p>
   </div>

   <div id="Paris" class="city" style="display:none">
   <h2>Paris</h2>
   <p>Paris is the capital of France.</p>
   </div>

   <div id="Tokyo" class="city" style="display:none">
   <h2>Tokyo</h2>
   <p>Tokyo is the capital of Japan.</p>
   </div>

   <script type = "text/javascript">  
      function openCity(cityName) {
      var i;
      var x = document.getElementsByClassName("city");
      for (i = 0; i < x.length; i++) {
         x[i].style.display = "none";
      }
      document.getElementById(cityName).style.display = "block";
      }
   </script>  


.. collapse:: Examples for Enterprises

   .. raw:: html

      <div class="display-card-container">
         <div class="row">

   .. displayitem::
      :header: Internal ML system (no UI)
      :description: An internal ML system for fraud detection.
      :col_css: col-md-4
      :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_full_control.png
      :button_link: https://lightning.ai/muse
      :height: 280
      :tag: Enterprise

   .. displayitem::
      :header: External SaaS product 
      :description: Build and monetize external products 
      :col_css: col-md-4
      :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_full_control.png
      :button_link: https://lightning.ai/muse
      :height: 280
      :tag: Enterprise

   .. displayitem::
      :header: Demo software products
      :description: Share reproducible software products for clients that scale instead of jupyter notebooks that don't.
      :col_css: col-md-4
      :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_full_control.png
      :button_link: https://lightning.ai/muse
      :height: 280
      :tag: Enterprise

   .. raw:: html

         </div>
      </div>

.. collapse:: Examples for Startups

   .. raw:: html

      <div class="display-card-container">
         <div class="row">

   .. displayitem::
      :header: SaaS product for generative AI
      :description: Launch and monetize a cloud SaaS products like this one.
      :col_css: col-md-4
      :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_full_control.png
      :button_link: https://lightning.ai/muse
      :height: 280
      :tag: Startups

   .. displayitem::
      :header: LLM app
      :description: Build and monetize external products 
      :col_css: col-md-4
      :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_full_control.png
      :button_link: https://lightning.ai/muse
      :height: 280
      :tag: Startups

   .. displayitem::
      :header: Demo software products
      :description: Share reproducible software products for clients that scale instead of jupyter notebooks that don't.
      :col_css: col-md-4
      :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_full_control.png
      :button_link: https://lightning.ai/muse
      :height: 280
      :tag: Startups

   .. raw:: html

         </div>
      </div>

.. collapse:: Examples for Research

   .. raw:: html

      <div class="display-card-container">
         <div class="row">

   .. displayitem::
      :header: Multi-node training
      :description: Product to ... 
      :col_css: col-md-4
      :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_full_control.png
      :button_link: https://lightning.ai/muse
      :height: 280
      :tag: Research

   .. displayitem::
      :header: LLM training
      :description: Build hyper-customized custom ML platforms. This one trains LLMs.
      :col_css: col-md-4
      :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_full_control.png
      :button_link: https://lightning.ai/muse
      :height: 280
      :tag: Research

   .. displayitem::
      :header: Visual demo with a public link
      :description: Create visual websites to demo models for quick POCs and demos in <1 hour.
      :col_css: col-md-4
      :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_full_control.png
      :button_link: https://lightning.ai/muse
      :height: 280
      :tag: Research

   .. raw:: html

         </div>
      </div>

.. collapse:: Examples for Hobbyists/Students

   .. raw:: html

      <div class="display-card-container">
         <div class="row">

   .. displayitem::
      :header: Cloud data scraper
      :description: An internal ML system for fraud detection.
      :col_css: col-md-4
      :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_full_control.png
      :button_link: https://lightning.ai/muse
      :height: 280
      :tag: Hobbyist or student

   .. displayitem::
      :header: Homework assignment
      :description: Build and monetize external products 
      :col_css: col-md-4
      :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_full_control.png
      :button_link: https://lightning.ai/muse
      :height: 280
      :tag: Hobbyist or student

   .. displayitem::
      :header: Cloud Jupyter Notebooks
      :description: Share reproducible software products for clients that scale instead of jupyter notebooks that don't.
      :col_css: col-md-4
      :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_full_control.png
      :button_link: https://lightning.ai/muse
      :height: 280
      :tag: Hobbyist or student

   .. raw:: html

         </div>
      </div>

.. note:: PyTorch or PyTorch Lightning knowledge is *NOT* required.
----

*************
Why Lightning
*************
Lightning provides a thin API that minimally **organizes Python code** to unlock modularity so you can build full stack AI applications ⚡ Lightning fast ⚡.

A 1-hour investment to learn the minimal Lightning API will save you 100s of hours of learning about kubernetes, fault-tolerance,
distributed programming, etc...

|

----

*************************
Step 1: Install Lightning
*************************
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

***************************
Step 2: Run any python code
***************************
Lightning organizes Python code. Drop ⚡ *any* ⚡ piece of code into the LightningWork class and run on the cloud or your own hardware:

.. raw:: html

    <div class="display-card-container">
        <div class="row">
        <div class="col-md-5">

Deploy this example:

.. join_slack::

[video showing this]

.. raw:: html

        </div>
        <div class="col-md-7">

.. code:: python
   :emphasize-lines: 4, 5 

   # app.py
   import lightning as L

   class LitWorker(L.LightningWork):
      def run(self):
         message = """
         ANY python code can run here such as:
            - train a model
            - launch a deployment server
            - label data
            - run a react.js, dash, streamlit, etc...
            - start a jupyter notebook
            - subprocess.Popen('echo run any shell script')"""
         print(message)

   app = L.LightningApp(LitWorker())

.. raw:: html

        </div>
        </div>
    </div>

**Lightning runs the same on the cloud and locally on your choice of hardware.**

Run on cloud machine in your own AWS account or fully-managed `Lightning cloud <https://lightning.ai/>`_:

.. code:: python

   lightning run app app.py --cloud

Run on your own hardware:

.. code:: python 
   
   lightning run app app.py

----

*******************************************
Run cloud agnostic and accelerator agnostic
*******************************************
Lightning decouples your code from the accelerators and cloud. To change the accelerator use **CloudCompute**:

.. raw:: html

    <div class="display-card-container">
        <div class="row">
        <div class="col-md-5">

[VIDEO SHOWING CODE]

.. raw:: html

        </div>
        <div class="col-md-7">


.. code:: python
   :emphasize-lines: 16

   # app.py
   import lightning as L

   class LitWorker(L.LightningWork):
      def run(self):
         message = """
         ANY python code can run here such as:
            - train a model
            - launch a deployment server
            - label data
            - run a react.js, dash, streamlit, etc...
            - start a jupyter notebook
            - subprocess.Popen('echo run any shell script')"""
         print(message)

   # run on 1 cloud GPU
   compute = L.CloudCompute("gpu")
   app = L.LightningApp(LitWorker(cloud_compute=compute))

.. raw:: html

        </div>
        </div>
    </div>

Run on a cloud GPU:

.. code:: python

   lightning run app app.py --cloud

Run on your own hardware:

.. code:: python

   lightning run app app.py

.. collapse:: Other supported accelerators

   |

   .. code:: python

      compute = L.CloudCompute('default')          # 1 CPU
      compute = L.CloudCompute('cpu-medium')       # 8 CPUs
      compute = L.CloudCompute('gpu')              # 1 T4 GPU
      compute = L.CloudCompute('gpu-fast-multi')   # 4 V100 GPU
      compute = L.CloudCompute('p4d.24xlarge')     # AWS instance name (8 A100 GPU)
      app = L.LightningApp(LitWorker(cloud_compute=compute))

   More machine types are available when you `run on your AWS account <??>`_.

.. collapse:: Run on your AWS account

   |
   .. include:: run_on_aws_account.rst

----

************
Key features
************
You now know enough to build simple AI applications. Here are a few key features available
to super-charge your work:

**Optimized hardware management:**

.. collapse:: Use a custom container
   
   |

   Run your cloud Lightning code with a custom container image by using **cloud_build_config**:

   .. code:: python 
      
      # use docker, gcp or any image provider
      cloud_config = L.BuildConfig(image="gcr.io/google-samples/hello-app:1.0")
      app = L.LightningApp(LitWorker(cloud_build_config=cloud_config))

.. collapse:: Auto-stop idle machines

   |

   **idle_timeout**: Turn off the machine when it's idle for n seconds.

   .. code:: python

      # IDLE TIME-OUT 

      # turn off machine when it's idle for 10 seconds
      compute = L.CloudCompute('gpu', idle_timeout=10)
      app = L.LightningApp(LitWorker(cloud_compute=compute))

   |

.. collapse:: Auto-timeout submitted work

   |
   **wait_timeout**: Wait n seconds for machine to be allocated by the cloud provider before cancelling the work:

   .. code:: python

      # WAIT TIME-OUT 
      
      # if the machine hasn't started after 60 seconds, cancel the work
      compute = L.CloudCompute('gpu', wait_timeout=60)
      app = L.LightningApp(LitWorker(cloud_compute=compute)

   |
   
.. collapse:: Use spot machines (~70% discount)

   |

   **spot**: Spot machines are ~70% cheaper because they are not guaranteed every time and can be turned off at any second without notice. Use spot for
   non critical or long-running workloads.

   .. code:: python

      # ask for a spot machine
      # wait 60 seconds before auto-switching to a full-priced machine
      compute = L.CloudCompute('gpu', spot=True, wait_timeout=60)
      app = L.LightningApp(LitWorker(cloud_compute=compute)

   |
   
|

**Optimized for massive data:**

.. collapse:: Work with massive datasets

   |

   A LightningWork might need a large working folder for certain workloads such as ETL pipelines, data collection, training models and processing datasets.

   Attach a disk up to 64 TB with **disk_size**:

   .. code:: python

      # use 100 GB of space on that machine (max size: 64 TB)
      compute = L.CloudCompute('gpu', disk_size=100)
      app = L.LightningApp(LitWorker(cloud_compute=compute)

   .. note:: when the work finishes executing, the disk will be deleted.

   |
   
.. collapse:: Mount cloud storage

   |

   To mount an existing s3 bucket, use **Mount**:

   .. code:: python
      :emphasize-lines: 9, 10

      # app.py
      import lightning as L

      class LitWorker(L.LightningWork):
         def run(self):
            os.listdir('/foo')
            file = os.file('/foo/cat.jpg')

      mount = L.Mount(source="s3://lightning-example-public/", mount_path="/foo")
      compute = L.CloudCompute(mounts=mount)
      app = L.LightningApp(LitWorker())

   Now use any library (like Python's os) to manage files:

   .. code:: python
      :emphasize-lines: 2, 7, 8

      # app.py
      import os
      import lightning as L

      class LitWorker(L.LightningWork):
         def run(self):
            os.listdir('/foo')
            file = os.file('/foo/cat.jpg')

      mount = L.Mount(source="s3://lightning-example-public/", mount_path="/foo")
      compute = L.CloudCompute(mounts=mount)
      app = L.LightningApp(LitWorker())

   .. note::

      To attach private s3 buckets, sign up for our early access: support@lightning.ai.

   |
   
|

**Production-ready:**

.. collapse:: Write systems not scripts or notebooks

   |

   Lightning is built to feel simple and like you are writing scripts,
   but you are implicitly building production-ready systems.

   |
   
.. collapse:: fault tolerant

   |

   ABC 

   |
   
.. collapse:: observable

   |

   ABC 

   |
   
.. collapse:: auto-scaled

   |

   ABC 

   |
   
.. collapse:: encrypted secrets

   |

   ABC 

   |
   
.. collapse:: SOC 2

   |

   ABC 

   |
   
----

*****************************
What does Lightning do for me
*****************************
**Packaged code:**

It guarantees that python code runs in any environment. The same code will run on your laptop, or any cloud
or private clusters. You don't have to think about the cluster or know anything about the cloud.

**Modular:**

Lightning allows you to incorporate multiple components together so you don't have to build each piece
of a system yourself. It's like javascript/react components for python.

**Rapid iteration:**

Iterate through ideas in hours not months because you don't have to learn a million other concepts

**Cost control:**

Lightning makes cloud code observable, easy to monitor, measures code in real-time and is super-optimized. 
All the optimizations we make under the hood, lower your cloud bill.
Machines can shut down or spin up faster. 

# show time spent vs yours in terms of cost saving

**Built-in guard rails:**

Code is built to be implicitly fault-tolerant, structured and minimizes room for error. Although it feels like you
are writing a python script, you are actually building a system. 

----

***************************
Use the community ecosystem
***************************

**Start from a template**:

The Lightning structure allows you to use self-contained components from the Lightning community
so you don't have to build every piece of functionality yourself. Check out our component gallery
for examples

----   

***************************
Next step: Build a workflow
***************************
In this simple example we ran one piece of Python code. In the next guide,
we'll learn to run multiple pieces of code together in a workflow. We call such a workflow, a *Lightning App*.

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Level 2: Build a workflow
   :description: Run multiple LightningWorks together 
   :col_css: col-md-12
   :button_link: build_a_machine_learning_workflow.html
   :height: 150
   :tag: beginner

.. raw:: html

        </div>
    </div>
