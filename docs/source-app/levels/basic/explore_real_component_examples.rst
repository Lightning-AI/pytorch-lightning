########################################
Level 3: Explore real component examples 
########################################
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
