#######################################
Communicate Between React and Lightning
#######################################
**Audience:** Anyone who wants to add a web user interface (UI) written in react to their app.

**pre-requisites:** Make sure you've already connected the React and Lightning app.

**Difficulty level:** intermediate.

----

************
Example code
************
To illustrate how to communicate between a React app and a lightning App, we'll be using the `example_app.py` file
which :doc:`lightning init react-ui <create_react_template>` created:

.. literalinclude:: ../../../../../src/lightning/app/cli/react-ui-template/example_app.py

and the App.tsx file also created by :doc:`lightning init react-ui <create_react_template>`:

.. literalinclude:: ../../../../../src/lightning/app/cli/react-ui-template/ui/src/App.tsx

----

******************************
Update React --> Lightning app
******************************
To change the Lightning app from the React app, use `updateLightningState`.

In this example, when you press **Start printing** in the React UI, it toggles
the `react_ui.vars.should_print`:

.. literalinclude:: ../../../../../src/lightning/app/cli/react-ui-template/ui/src/App.tsx
    :emphasize-lines: 20, 21, 23

By changing that variable in the Lightning app state, it sets **react_ui.should_print** to True, which enables the
Lightning app to print:

.. literalinclude:: ../../../../../src/lightning/app/cli/react-ui-template/example_app.py
    :emphasize-lines: 10, 22

----

******************************
Update React <-- Lightning app
******************************
To change the React app from the Lightning app, use the values from the `lightningState`.

In this example, when the ``react_ui.counter`` increaes in the Lightning app:

.. literalinclude:: ../../../../../src/lightning/app/cli/react-ui-template/example_app.py
    :emphasize-lines: 18, 24

The React UI updates the text on the screen to reflect the count

.. literalinclude:: ../../../../../src/lightning/app/cli/react-ui-template/ui/src/App.tsx
    :emphasize-lines: 15
