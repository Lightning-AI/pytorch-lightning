:orphan:

##################################
Integrate any javascript framework
##################################
**Audience:** Advanced web developers with complex apps that may not have been covered by the other tutorials

**Pre-requisites:** Intermediate knowledge of html and javascript

----

************************
Import LightningState.js
************************
To connect any javascript framework, import the `LightningState.js <https://storage.googleapis.com/grid-packages/lightning-ui/v0.0.0/LightningState.js>`_ library.
LightningState.js enables two-way communication between a javascript framework and a Lightning app.

To import this library, add this to your html:

.. code:: html

    <script src="https://storage.googleapis.com/grid-packages/lightning-ui/v0.0.0/LightningState.js"></script>

Once it's imported, use it inside your app, this example uses it inside a React App:

.. code-block::
    :emphasize-lines: 1, 5

        import { useLightningState } from "./hooks/useLightningState";
        import cloneDeep from "lodash/cloneDeep";

        function App() {
            const { lightningState, updateLightningState } = useLightningState();

            const modify_and_send_back_the_state = async (event: ChangeEvent<HTMLInputElement>) => {
                if (lightningState) {
                const newLightningState = cloneDeep(lightningState);
                // Update the state and send it back.
                newLightningState.flows.counter += 1

                updateLightningState(newLightningState);
                }
        };

        return (
            <div className="App">
            </div>
        );
    }

    export default App;

----

************************
Update the Lightning app
************************
Use `updateLightningState` to update the lightning app. Here we update a variable called counter.

.. code-block::
    :emphasize-lines: 11

    import { useLightningState } from "./hooks/useLightningState";
    import cloneDeep from "lodash/cloneDeep";

    function App() {
            const { lightningState, updateLightningState } = useLightningState();

            const modify_and_send_back_the_state = async (event: ChangeEvent<HTMLInputElement>) => {
                if (lightningState) {
                const newLightningState = cloneDeep(lightningState);
                // Update the state and send it back.
                newLightningState.flows.counter += 1

                updateLightningState(newLightningState);
                }
        };

        return (
            <div className="App">
            </div>
        );
    }

    export default App;

----

**************************************
Receive updates from the Lightning app
**************************************
Whenever a variable in the Lightning app changes, the javascript app will receive those values via `lightningState`.

Extract any variable from the state and update the javascript app:

.. code-block::
    :emphasize-lines: 5

    import { useLightningState } from "./hooks/useLightningState";
    import cloneDeep from "lodash/cloneDeep";

    function App() {
            const { lightningState, updateLightningState } = useLightningState();

            const modify_and_send_back_the_state = async (event: ChangeEvent<HTMLInputElement>) => {
                if (lightningState) {
                const newLightningState = cloneDeep(lightningState);
                // Update the state and send it back.
                newLightningState.flows.counter += 1

                updateLightningState(newLightningState);
            }
        };

        return (
            <div className="App">
            </div>
        );
    }

    export default App;

----

********
Examples
********

See this in action in these examples:


.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
    :header: React.js
    :description: Explore how React.js uses lightningState.js
    :col_css: col-md-4
    :button_link: react/communicate_between_react_and_lightning.html
    :height: 150
    :tag: intermediate

.. displayitem::
    :header: Example 2
    :description: Show off your work! Contribute an example.
    :col_css: col-md-4
    :height: 150
    :tag: Waiting for contributed example

.. displayitem::
    :header: Example 3
    :description: Show off your work! Contribute an example.
    :col_css: col-md-4
    :height: 150
    :tag: Waiting for contributed example

.. raw:: html

        </div>
    </div>
