#######################
Lightning in 15 minutes
#######################

**Minimal CS background:** You know *basic* Python.

**Miniam ML background:** None! (PyTorch or PyTorch Lightning knowledge is not required).

**Goal:** In this guide you'll learn the basic concept to develop with Lightning.

.. join_slack::
   :align: left

----

******************
What is Lightning?
******************
Lightning is an `open-source <https://github.com/Lightning-AI/lightning>`_ framework that provides **minimal organization to Python code** that connects `community-built components <https://lightning.ai/components>`_ to develop workflows that
`run on your own AWS account <#run->`_, the `Lightning Cloud (fully-managed AWS) <https://lightning.ai/>`_ or `your own hardware <?>`_.

|

Examples of what you can build with Lightning:

.. collapse:: For Enterprises

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

.. collapse:: For Startups

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

.. collapse:: For Research

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

|

⚡⚡ This guide will teach you the main principles that allow you to build systems like the ones above Lightning fast ⚡⚡.

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

**Community-built components:**

Build with the community

----

*************************
Step 1: Install Lightning
*************************
.. code:: bash

    python -m pip install -U lightning

----

***************************
Step 2: Run any python code
***************************
[video showing this]

Lightning organizes Python code. Drop any piece of code into the LightningWork class and run on the cloud or your own hardware:

.. code:: python

   # app.py
   import lightning as L

   class LitWorker(L.LightningWork):
      def run(self):
         message = """
         ANY python code can run here such as:
         - train a model
         - launch a deployment server
         - label data
         - run a react app, dash app, streamlit app, etc...
         - start a jupyter notebook
         - subprocess.Popen('echo run any shell script, python scripts or non python files')
         """
         print(message)

   # uses 1 cloud GPU (or your own hardware)
   compute = L.CloudCompute('gpu')
   app = L.LightningApp(LitWorker(cloud_compute=compute))


**Lightning runs the same on the cloud and locally.**

Run on a GPU in your own AWS account or Lightning Cloud (fully-managed AWS):

.. code:: python

   lightning run app.py --cloud

Run on your own hardware:

.. code:: python 
   
   lightning run app.py

----

************
Key features
************
You now know enough to build pretty powerful cloud workflows. Here are some features available
to super-charge your work.

**Cloud and hardware agnostic:**

.. collapse:: Use different cloud accelerators

   |

   .. code:: python

      compute = L.CloudCompute('default')          # 1 CPU
      compute = L.CloudCompute('cpu-medium')       # 8 CPUs
      compute = L.CloudCompute('gpu')              # 1 T4 GPU
      compute = L.CloudCompute('gpu-fast-multi')   # 4 V100 GPU
      compute = L.CloudCompute('p4d.24xlarge')     # AWS instance name (8 A100 GPU)
      app = L.LightningApp(LitWorker(cloud_compute=compute))

   More machine types are available when you `run on your AWS account <??>`_.

   |

.. collapse:: Use a custom container
   
   |

   Run your cloud Lightning code with a custom container image by using **cloud_build_config**:

   # TODO: only google?

   .. code:: python 
      
      # USE A CUSTOM CONTAINER

      cloud_config = L.BuildConfig(image="gcr.io/google-samples/hello-app:1.0")
      app = L.LightningApp(LitWorker(cloud_build_config=cloud_config))

   |

.. collapse:: Run on your AWS account

   |
   To run on your own AWS account, first `create an AWS ARN <../glossary/aws_arn.rst>`_.   

   Next, set up a Lightning cluster (here we name it pikachu):

   .. code:: bash

      # TODO: need to remove  --external-id dummy --region us-west-2
      lightning create cluster pikachu --provider aws --role-arn arn:aws:iam::1234567890:role/lai-byoc

   Run your code on the pikachu cluster by passing it into CloudCompute:

   .. code:: python 

      compute = L.CloudCompute('gpu', clusters=['pikachu'])
      app = L.LightningApp(LitWorker(cloud_compute=compute))

   .. warning:: 
      
      This feature is available only under early-access. Request access by emailing upgrade@lightning.ai.

   |

|

**Optimized hardware management:**

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
   
.. collapse:: Use preemptible machines (~70% discount)

   |
   **preemptible**: Pre-emptible machines are ~70% cheaper because they can be turned off at any second without notice:

   .. code:: python
      
      # PRE-EMPTIBLE MACHINES

      # ask for a preemptible machine
      # wait 60 seconds before auto-switching to a full-priced machine
      compute = L.CloudCompute('gpu', preemptible=True, wait_timeout=60)
      app = L.LightningApp(LitWorker(cloud_compute=compute)

   |
   
|

**Optimized for massive data:**

.. collapse:: Work with massive datasets

   |

   A LightningWork might need a large working folder for certain workloads such as ETL pipelines, data collection, training models and processing datasets.

   Attach a disk up to 64 TB with **disk_size**:

   .. code:: python

      # MODIFY DISK SIZE 

      # use 100 GB of space on that machine (max size: 64 TB)
      compute = L.CloudCompute('gpu', disk_size=100)
      app = L.LightningApp(LitWorker(cloud_compute=compute)

   .. note:: when the work finishes executing, the disk will be deleted.

   |
   
.. collapse:: Mount cloud storage

   |

   To mount an existing s3 bucket, use **Mount**:

   .. code:: python

      # TODO: create a public demo folder
      # public bucket
      mount = Mount(source="s3://lightning-example-public/", mount_path="/foo")
      compute = L.CloudCompute(mounts=mount)

      app = L.LightningApp(LitWorker(cloud_compute=compute))

   Read and list the files inside your LightningWork:

   .. code:: python

      # app.py
      import lightning as L

      class LitWorker(L.LightningWork):
         def run(self):
            os.listdir('/foo')
            file = os.file('/foo/a.jpg')

      app = L.LightningApp(LitWorker())

   .. note::

      To attach private s3 buckets, sign up for our early access: support@lightning.ai.

   |
   
|

**Community-driven:**

.. collapse:: Use community-built LightningWorks

   |

   The Lightning structure allows you to use self-contained components from the Lightning community
   so you don't have to build every piece of functionality yourself. Check out our component gallery
   for examples

   |
   
.. collapse:: Learn and get help

   |

   Over 400k people across the world build with Lightning. Join our community to learn from the best, ask any questions
   or just hang out!

   .. join_slack::
      :align: center

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

***************************
Next step: Build a workflow
***************************
In this simple example we ran one piece of Python code. To create a complex workflow easily,
we'll need to learn how to use multiple works together.


.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Next step: Build a workflow
   :description: Run multiple LightningWorks together 
   :col_css: col-md-12
   :button_link: ../model/build_model_advanced.html#manual-optimization
   :height: 150
   :tag: beginner

.. raw:: html

        </div>
    </div>
