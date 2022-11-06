###############################################
Level 2: Explore real component implementations
###############################################
**Audience:** Users who want to deeply understand what is possible with Lightning components.

**Prereqs:** You must have finished `level 1 <../basic/build_a_lightning_component.html>`_, `level 2 <../basic/debug_a_lightning_workflow.html>`_.

----

***************************
Debug a lightning component
***************************
Before we dive into real component implementations, we'll learn to debug a Lightning component.

To stop the code execution at a particular line, enable a breakpoint
with **L.pdb.set_trace()**:

.. lit_tabs::
   :descriptions: Toy app; Add a breakpoint. When the program runs, it will stop at this line.
   :code_files: ./scripts/toy_app_1_component.py; ./scripts/toy_app_1_component_pdb.py
   :highlights: ; 7
   :app_id: abc123
   :tab_rows: 3
   :height: 350px

|

.. note:: 

   Direct python pdb support is work in progress and open to contributions

----

*****************************
Ex 1: Train PyTorch component
*****************************
TODO:

----

******************************
Next: Coordinate 2+ components
******************************
Now that you know how to organize arbitrary code inside a Lightning component,
learn to coordinate 2 or more components into workflows which we call Lightning apps. 

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Intermediate skills
   :description: Learn to coordinate 2+ components into workflows which we call Lightning apps.
   :button_link: ../intermediate/index.html
   :col_css: col-md-12
   :height: 170
   :tag: 15 minutes

.. raw:: html

        </div>
    </div>
