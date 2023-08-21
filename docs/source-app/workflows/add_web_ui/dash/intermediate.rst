#####################################
Add a web UI with Dash (intermediate)
#####################################
**Audience:** Users who want to communicate between the Lightning App and Dash.

**Prereqs:** Must have read the :doc:`dash basic <basic>` guide.

----

*******************************
Interact with the App from Dash
*******************************

In the example below, every time you change the select year on the dashboard, this is directly communicated to the flow
and another work process the associated data frame with the provided year.

.. literalinclude:: intermediate_plot.py

Here is how the app looks like once running:

.. figure::  https://pl-public-data.s3.amazonaws.com/assets_lightning/dash_plot.gif

----

***********************************
Interact with Dash from a component
***********************************

In the example below, when you click the toggle, the state of the work appears.

Install the following libraries if you want to run the app.

```bash
pip install dash_daq dash_renderjson
```

.. literalinclude:: intermediate_state.py


Here is how the app looks like once running:

.. figure::  https://pl-public-data.s3.amazonaws.com/assets_lightning/dash_state.gif
