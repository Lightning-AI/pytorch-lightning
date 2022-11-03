##############################
Level 4: Debug A Lightning app
##############################
**Audience:** Users who want to debug a distributed app locally.

**Prereqs:** You must have finished the `Basic levels <../basic/>`_.

----

**********************
Enable local debugging
**********************
A distributed Lightning app can run locally on a single machine. To debug apps running locally
you can use the multi-processing runtime:

.. lit_tabs::
   :titles: Toy app; Enable MultiProcessRuntime
   :code_files: ./scripts/toy_app.py; ./scripts/debug_app.py
   :highlights: ; 3, 24
   :app_id: abc123
   :tab_rows: 3
   :height: 480px
