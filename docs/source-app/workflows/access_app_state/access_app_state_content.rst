*******************
A bit of background
*******************

Lightning allows you to create reactive distributed applications, where components can be distributed across different machines and even different clouds.

To create reactive applications, components need to be able to communicate- share data, status, values, etc. For example, if you are creating an app that will retrain a model every time new data is added to a cloud dataset, you will need the dataset component to communicate to the training component.

Lightning components can communicate via the app state. The app state is composed of all attributes defined within each components **__init__** method.

All attributes of all LightningWork components are accessible in the LightningFlow components in real time.

----

*******************************
What the App State does for you
*******************************

Every time you update an attribute inside of a running work (separate process on local or remote machine), the attribute of the mirrored work in the flow side is updated with that same exact value automatically.

Every time the app received a state update from a running work, the app applies the state change and re-executes the flow run method. This enables complex systems to be easily implemented through the state.

The **App State** is the collection of all the components state forming the application and gets automatically up-to-date, even in distributed settings.

----

********************
Access the App State
********************

As a user, you are interacting with your component attributes, so most likely,
you won't need to access the component state directly, but it can be helpful to
understand how the state works under the hood.

For example, here we define a **Flow** component and **Work** component, where the work increments a counter indefinitely and the flow prints its state which contains the work.

You can easily check the state of your entire app as follows:

.. literalinclude:: ../../workflows/access_app_state/app.py

Run the app with:

.. code-block:: bash

    lightning run app docs/source-app/workflows/access_app_state/app.py

And here's the output you get when running the above application using **Lightning CLI**:

.. code-block:: console

	INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
	State: {'works': {'w': {'vars': {'counter': 1}}}}
	State: {'works': {'w': {'vars': {'counter': 2}}}}
	State: {'works': {'w': {'vars': {'counter': 3}}}}
	State: {'works': {'w': {'vars': {'counter': 3}}}}
	State: {'works': {'w': {'vars': {'counter': 4}}}}
	...


***
FAQ
***

* **How can a work update a flow ?** A work is a leaf in the component tree, therefore it can access / update only itself.

* **How can a flow update a work ?** The flow can update the work state. However, once the run method is launched, the work is running isolated with a copy of state. If the flow keeps updating the work state, and the work does the same with different values, it creates a state divergence. In the future, we might support bi-directional state update between flow and works.

* **How can a work update a work ?** No, the communication is simply between work and flow. The flow role is to coordinate and pass some state between works.

* **How can a flow update a flow ?** Yes, the flows can update themselves as they are running in the same python process.
