##############################
Level 5: Debug A Lightning app
##############################
**Audience:** Users who want to debug a distributed app locally.

**Prereqs:** You must have finished the :doc:`Basic levels <../basic/index>`.

----

******************
Enable breakpoints
******************
To enable a breakpoint, use :func:`~lightning.app.pdb.set_trace()` (note direct python pdb support is work in progress and open to contributions).

.. lit_tabs::
   :descriptions: Toy app; Add a breakpoint. When the program runs, it will stop at this line.
   :code_files: ./debug_app_scripts/toy_app_1_component.py; ./debug_app_scripts/toy_app_1_component_pdb.py
   :highlights: ; 7
   :enable_run: true
   :tab_rows: 3
   :height: 350px

----

*********************************
Next: Run a component in parallel
*********************************
Learn to run components in parallel to enable more powerful workflows.

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Level 6: Run a Lightning component in parallel
   :description: Learn when and how to run Components in parallel (asynchronous).
   :button_link: run_lightning_work_in_parallel.html
   :col_css: col-md-12
   :height: 150
   :tag: 15 minutes

.. raw:: html

        </div>
    </div>
