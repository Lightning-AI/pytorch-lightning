.. _access_app_state:

################
Access App State
################

**Audience:** Users who want to know how the app state can be accessed.

**Level:** Basic

***********************
What is the App State ?
***********************

In Lightning, each component is stateful and their state are composed of all attributes defined within their **__init__** method.

The **App State** is the collection of all the components state forming the application.

*****************************************
What is the special about the App State ?
*****************************************

The **App State** is always up-to-date, even running an application in the cloud on multiple machines.

This means that every time an attribute is modified in a work, that information is automatically broadcasted to the flow.

With this mechanism, any component can **react** to any other component **state changes** through the flow and complex system can be easily implemented.

Lightning requires a state based driven mindset when implementing the flow.

************************************
When do I need to access the state ?
************************************

As a user, you are interacting with your component attributes, so most likely,
you won't need to access the component state directly, but it can be helpful to
understand how the state works under the hood.

For example, here we define a **Flow** component and **Work** component, where the work increments a counter indefinitely and the flow prints its state which contains the work.

You can easily check the state of your entire app as follows:

.. literalinclude:: ../code_samples/quickstart/app_01.py

Run the app with:

.. code-block:: bash

    lightning run app docs/quickstart/app_01.py

And here's the output you get when running the above application using **Lightning CLI**:

.. code-block:: console

	INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
	State: {'works': {'w': {'vars': {'counter': 1}}}}
	State: {'works': {'w': {'vars': {'counter': 2}}}}
	State: {'works': {'w': {'vars': {'counter': 3}}}}
	State: {'works': {'w': {'vars': {'counter': 3}}}}
	State: {'works': {'w': {'vars': {'counter': 4}}}}
	...
