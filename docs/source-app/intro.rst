:orphan:

.. _what:

###################
What is Lightning?
###################

Lightning is a free, modular, distributed, and open-source framework for building
AI applications where the components you want to use interact together.

Lightning apps can be built for **any AI use case**, ranging from AI research to
production-ready pipelines (and everything in between!).

By abstracting the engineering boilerplate, Lightning allows researchers, data scientists, and software engineers to
build highly-scalable, production-ready AI apps using the tools and technologies of their choice,
regardless of their level of engineering expertise.

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/Lightning.gif
    :alt: What is Lightning gif.
    :width: 100 %

----

.. _why:

***************
Why Lightning?
***************


Easy to learn
^^^^^^^^^^^^^

Lightning was built for creating AI apps, not for dev-ops. It offers an intuitive, pythonic
and highly composable interface that allows you to focus on solving the problems that are important to you.

----

Quick to deliver
^^^^^^^^^^^^^^^^

Lightning speeds the development process by offering testable templates you can build from,
accelerating the process of moving from idea to prototype and finally to market.

----

Easy to scale
^^^^^^^^^^^^^

Lightning provides a mirrored experience locally and in the cloud. The `lightning.ai <https://lightning.ai>`_.
cloud platform abstracts the infrastructure, so you can run your apps at any scale.

----

Easy to collaborate
^^^^^^^^^^^^^^^^^^^

Lightning was built for collaboration.
By following the best MLOps practices provided through our documentation and example use cases,
you can deploy state-of-the-art ML applications that are ready to be used by teams of all sizes.

----

*****************************
What's Novel With Lightning?
*****************************


Cloud Infra Made Simple and Pythonic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Lightning is for building reactive, scalable, cost effective, easy-to-maintain and reliable ML products in the cloud without worrying about infrastructure. Lightning provides several engineering novelties to enable this:

#. **Reactivity**: Lightning allows you to run stateful components distributed across different machines, so you can design async, dynamic and reactive workflows in python, without having to define DAGs.

#. **Scalable & Cost-Effective**: Lightning provides a granular and simple way to run components preemptively or on-demand and on any desired resource such as CPU or GPU. It also enables you to easily transfer artifacts from one machine to another.

#. **Reliability**:

    #. **Checkpointing**: Lightning apps can be paused and resumed from generated state and artifact-based checkpoints.
    #. **Resilience**: Lightning has a strong fault-tolerance foundation. Your application can be written and tested to be resilient for cloud hazards at the component level.
    #. **Testing Tools**: Lightning provides you with tools and best practices you can use to develop and test your application. All of our built-in templates have unit integration and end-to-end tests.

#. **Easy to maintain**:

    #. **Easy Debugging**: Lightning apps can be debugged locally and in the cloud with **breakpoints** in any components.
    #. **Non-Invasive**: Lightning is the glue that connects all parts of your workflow, but this is done in a non-invasive way by formalizing API contracts between components. In other words, your application can run someone else's code with little assumption.
