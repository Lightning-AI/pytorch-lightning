##############################
Level 2: Debug A Lightning app
##############################
**Audience:** Users who want to debug a distributed app locally.

**Prereqs:** You must have finished the `Basic levels <../basic/>`_.

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
