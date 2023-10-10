######################################
Create a React Template (intermediate)
######################################
**Audience:** Anyone who wants to add a web user interface (UI) written in react to their app.

----

**************
What is react?
**************
`React.js <https://reactjs.org/>`_ is a JavaScript library for building user interfaces.
A huge number of websites are written in React.js (like Facebook).

----

************************
Bring your own React app
************************
If you already have a React.js app, then you don't need the section below. However, it might be helpful
to see our React template so you can understand how to connect it to a Lightning app.

----

****************************
Create the react-ui template
****************************
Lightning can generate a react-ui template out of the box (generated with `Vite <https://github.com/vitejs/vite>`_).

Run this command to set up a react-ui template for a component:

.. code:: bash

    lightning init react-ui

If everything was successful, run the example_app.py listed in the output of the command:

.. code:: bash

    INFO: Checking pre-requisites for react
    INFO:
        found npm  version: 8.5.5
        found node version: 16.15.0
        found yarn version: 1.22.10

    ...
    ...

    ⚡ run the example_app.py to see it live!
    lightning run app react-ui/example_app.py

If the command didn't work, make sure to install `npm+nodejs <https://docs.npmjs.com/downloading-and-installing-node-js-and-npm>`_, and `yarn <https://classic.yarnpkg.com/lang/en/docs/install/#mac-stable>`_.
