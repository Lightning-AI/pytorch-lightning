


***************************************
When to run a LightningWork in parallel
***************************************
Run LightningWork in parallel when you want to execute work in the background or at the same time as another work.
An example of when this comes up in machine learning is when data streams-in while a model trains.

----

********************
Run work in parallel
********************
By default, a LightningWork must complete before the next one runs:

.. lit_tabs::
   :descriptions: Toy app; Run the train component in parallel so we can immediately start analysis without waiting for A to complete; Train and baseline in parallel which launches analysis immediately.
   :code_files: /workflows/scripts/parallel/toy_app.py; /workflows/scripts/parallel/toy_parallel.py; /workflows/scripts/parallel/toy_two_parallel.py
   :highlights: ; 17; 17, 18, 22, 23
   :app_id: abc123
   :tab_rows: 3
   :height: 520px
