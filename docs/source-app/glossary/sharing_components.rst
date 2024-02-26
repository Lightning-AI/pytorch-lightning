#####################
Sharing my components
#####################

**Audience:** Users who want to know how to share component.

**Level:** Basic

----

********************************************
Why should I consider sharing my components?
********************************************

Lightning is community driven and its core objective is to make AI accessible to everyone.

By creating components and sharing them with everyone else, the barrier to entry will go down.

----

************************************
How should I organize my components?
************************************

By design, Lightning components are nested to form component trees where the ``LightningFlows`` are its branches and ``LightningWorks`` are its leaves.

This design has two primary advantages:

* This helps users organize and maintain their code with more ease.
* This also helps create an ecosystem with **reusable** components.


Now, imagine you have implemented a **KerasScriptRunner** component for training any `Keras <https://github.com/keras-team/keras>`_ model with `Tensorboard UI <https://github.com/tensorflow/tensorboard>`_ integrated.

Here are the best practices steps before sharing the component:

* **Testing**: Ensure your component is well tested by following the :doc:`../testing` guide.
* **Documented**: Ensure your component has a docstring and comes with some usage explications.

.. Note:: As a Lightning user, it helps to implement your components thinking someone else is going to use them.

----

*****************************************
How should I proceed to share components?
*****************************************

Once your component is ready, create a *PiPy* package with your own library and then it can be reused by anyone else.

Here is a `Component Template <https://github.com/Lightning-AI/LAI-slack-messenger>`_ from `William Falcon <https://www.williamfalcon.com/>`_ to guide your component.
