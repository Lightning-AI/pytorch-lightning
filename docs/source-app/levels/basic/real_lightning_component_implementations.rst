###############################################
Level 2: Explore real component implementations
###############################################
**Audience:** Users who want to deeply understand what is possible with Lightning components.

**Prereqs:** You must have finished `level 1 <../basic/build_a_lightning_component.html>`_, `level 2 <../basic/debug_a_lightning_workflow.html>`_.

----

******************
Enable breakpoints
******************
To enable a breakpoint, use `L.pdb.set_trace()` (note direct python pdb support is work in progress and open to contributions).

.. lit_tabs::
   :descriptions: Toy app; Add a breakpoint. When the program runs, it will stop at this line.
   :code_files: ./scripts/toy_app_1_component.py; ./scripts/toy_app_1_component_pdb.py
   :highlights: ; 7
   :app_id: abc123
   :tab_rows: 3
   :height: 350px

----

************************
Next: Connect components
************************
Now that you know how to organize arbitrary code inside a Lightning component,
learn to coordinate 2 or more components into workflows which we call Lightning apps. 

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Level 3: Connect components into a full stack AI app
   :description: Learn to connect components into a full stack AI app.
   :button_link: ../intermediate/connect_lightning_components.html
   :col_css: col-md-12
   :height: 170
   :tag: 15 minutes

.. raw:: html

        </div>
    </div>
