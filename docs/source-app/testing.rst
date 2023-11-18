:orphan:

.. _testing:

#######################
Productionize your Apps
#######################

.. TODO: Cleanup

At the core of our system is an integration testing framework that will allow for a first-class experience creating integration tests for Lightning Apps. This document will explain how we can create a lightning app test, how we can execute it, and where to find more information.

----

***********
Philosophy
***********

Testing a Lightning app is unique. It is a superset of an application that converges machine learning, API development, and UI development. With that in mind, there are several philosophies (or "best practices") that you should adhere to:


#. **Control your app state** - With integration and end to end tests, you have the capabilities of controlling your app's state through dependency injection. Use it!
#. **Integration focuses on the work, End to End focuses on the app** - When writing tests, think of the depth and breath of what you are writing. Write many integration tests since they are relatively cheap, while keeping the end to end tests for holistic app testing.
#. **Don't overthink it** - What needs to be tested? What is the order of risk? These are the questions you should build with before writing your first line of code. Writing tests for the sake of writing tests is an exercise in futility. Write meaningful, impactful tests.
#. **Test Isolation** - Write your tests in an isolated manner. No two tests should ever depend on each other.
#. **Use your framework** - Testing apps should be framework agnostic.
#. **Have fun!** - At the heart of testing is experimentation. Like any experiment, tests begin with a hypothesis of workability, but you can extend that to be more inclusive. Ask the question, write the test to answer your question, and make sure you have fun while doing it.

----

****************************************
Anatomy of a Lightning integration test
****************************************

The following is a PyTest example of an integration test using the ``lightning.app.testing`` module.

.. code-block:: python

   import os

   from lightning.app import _PROJECT_ROOT
   from lightning.app.testing import application_testing, LightningTestApp
   from lightning.app.utilities.enum import AppStage


   class TestLightningAppInt(TestLightningApp):
       def run_once(self) -> bool:
           if self.root.counter > 1:
               print("V0 App End")
               self.stage = AppStage.STOPPING
               return True
           return super().run_once()


   def test_v0_app_example():
       command_line = [
           os.path.join(_PROJECT_ROOT, "examples/app_v0/app.py"),
           "--blocking",
           "False",
           "--multiprocess",
           "--open-ui",
           "False",
       ]
       result = application_testing(TestLightningAppInt, command_line)
       assert "V0 App End" in str(result.stdout_bytes)
       assert result.exit_code == 0

----

Setting up the app
^^^^^^^^^^^^^^^^^^

Lightning apps are unique in that they represent a full stack model for your machine learning application. To be clear, the integration tests are *NOT* going to touch the UI flow. Instead we inject your application with helper methods that, when executed, can assist in validating your application.

To get started, you simply need to import the following:

.. code-block:: python

    from lightning.app.testing import application_testing, LightningTestApp

We will discuss ``application_testing`` in a bit, but first let's review the structure of ``LightningTestApp``.

----

LightningTestApp
^^^^^^^^^^^^^^^^^

The :class:`lightning.app.testing.testing.LightningTestApp` class is available to use for provisioning and setting up your testing needs. Note that you do not need this class to move forward with testing. Any application that inherits ``LightningApp`` should suffice as long as you override the correct methods. Reviewing the TestLightnigApp we see some overrides that are already there. Please revuew the class for more information.

.. code-block:: python

   class TestLightningAppInt(LightningTestApp):
       def run_once(self) -> bool:
           if self.root.counter > 1:
               print("V0 App End")
               self.stage = AppStage.STOPPING
               return True
           return super().run_once()

We create a test class overriding the ``run_once`` function. This function helps control the flow of your application and is ran first. In this example we are calling ``self.root.counter`` and checking if the job has executed once. If so, we want to print out ``V0 App End`` and set the ``self.stage`` to ``AppStage.STOPPING``. This is how we control the flow through state. Your situation might be different, so experiment and see what you can do!

Besides ``run_once`` there are a few other overrides available:


* ``on_before_run_once`` - This runs before your ``run_once`` function kicks off. You can set up your application pre-conditions here.
* ``on_after_run_once`` - Similar to ``on_before_run_once`` but after the ``run_once`` method is called.

These methods will skew your tests, so use them when needed.

----

The Test
^^^^^^^^

We provide ``application_testing`` as a helper function to get your application up and running for testing. It uses ``click``\ 's invocation tooling underneath.

.. code-block::

   command_line = [
       os.path.join(_PROJECT_ROOT, "examples/app_v0/app.py"),
       "--blocking",
       "False",
       "--open-ui",
       "False",
   ]

First in the list for ``command_line`` is the location of your script. It is an external file. In this example we have ``_PROJECT_ROOT`` but this is *not* a helper constant for you to utilize. You will need to provide the location yourself.

Next there are a couple of options you can leverage:

* ``blocking`` - Blocking is an app status that says "Do not run until I click run in the UI". For our integration test, since we are not using the UI, we are setting this to "False".
* ``open-ui`` - We set this to false since this is the routine that opens a browser for your local execution.

Once you have your commandline ready, you will then be able to kick off the test and gather results:

.. code-block:: python

   result = application_testing(TestLightningAppInt, command_line)

As mentioned earlier, ``application_testing`` is a helper method that allows you to inject your TestLightningApp class (with overrides) and the commandline flags. Once the process is done it returns the results back for parsing.

.. code-block:: python

   assert "V0 App End" in str(result.stdout_bytes)
   assert result.exit_code == 0

Since we injected "V0 App End" to the end of our test flow. The state was changed to ``AppStatus.STOPPING`` which means the process is done. Finally, we check the result's exit code to make sure that we did not throw an error during execution.

----

************
End to End
************

TODO
