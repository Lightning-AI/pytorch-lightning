.. toctree::
   :maxdepth: 1

################################
New to Lightning? READ ME FIRST!
################################

Lightning is all about the APPS! THE LIGHTNING APPS!

The Lightning framework is a distributed, modular, free, and open framework you can use to build all AI applications (Lightning Apps) where the components you want, interact together.

Use the Lightning framework to build anything from production-ready, multi-cloud ML systems to simple research demos.

With the framework you can run and even train your App models on premise or in the cloud on various CPU and GPU hardware.

You can create your Apps from scratch, build them from a template, OR check out the `Apps Gallery <https://lightning.ai/apps>`_ and
clone and then modify existing Apps for your own use.

You can also build components for your Apps from scratch, from a template, OR use existing components from the `Components Gallery <https://lightning.ai/components>`_.

----

**************************
I'm totally new (to AI/ML)
**************************

No worries! Just `install Lightning <installation.rst>`_ and then `clone and run Flashy <https://lightning.ai/app/PgM82rHUWu-Flashy>`_!
Flashy will have you run your first ML models in no time, with no coding!

While Flashy is the fastest way to get started with Machine Learning, eventually you'll want
to create your own Lightning Apps. When you're ready you can clone Apps and components
from the Apps and `App Gallery <https://lightning.ai/apps>`_ (then modify them), build Apps or components
from templates, or build Apps and components from scratch.

*********************************************************
I have some XP in AI/ML. LET'S BUILD SOME LIGHTNING APPS!
*********************************************************

You have maybe built some demos or some enterprise production-ready ML Solutions, just not with Lightning.

Let's get you started. Here's a few important things to keep in mind when building your Lightning Apps (App):

* Always check the `Apps Gallery <https://lightning.ai/apps>`_ and `Apps Components <https://lightning.ai/components>`_ first. Why build something from scratch when you can clone an App or add a component and then build off of someone else's work.
* You can run your Apps locally or in the cloud on CPUs or GPUs. To get started, you can only run one App locally.
* Lightning Apps consist of a `root LightningFlow component <https://lightning.ai/lightning-docs/lightning_apps_intro.html#define-root-component>`_, that optionally contains a tree of 2 types of components: `LightningFlow <https://lightning.ai/lightning-docs/core_api/lightning_flow.html>`_ 🌊 and `LightningWork <https://lightning.ai/lightning-docs/core_api/lightning_work/>`_ ⚒️. Key functionality includes:

  * :ref:`A shared state between components. <app_component_tree>`
  * :ref:`A constantly running event loop for reactivity. <app_event_loop>`
  * :ref:`Dynamic attachment of components at runtime. <dynamic_work>`
  * Start and stop functionality of your works.


If you're ready to go, here's what you're gonna wanna do:

#. `Install Lightning <installation.rst>`_
#. Check out the `Lightning in 15 minutes <lightning_apps_intro.rst>`_ topic.
#. Start learning those Lightning App Building Skills!

   * Start with Lightning's `Basic Skills <https://lightning.ai/lightning-docs/levels/basic/>`_.
   * Move on to `Intermediate Skills <https://lightning.ai/lightning-docs/levels/intermediate/>`_.
   * And when you're ready the `Advanced Skills <https://lightning.ai/lightning-docs/levels/advanced/>`_.
