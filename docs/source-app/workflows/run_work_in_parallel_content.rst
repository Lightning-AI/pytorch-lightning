


************************************
When to run a Components in parallel
************************************
Run LightningWork in parallel when you want to execute work in the background or at the same time as another work.
An example of when this comes up in machine learning is when data streams-in while a model trains.

----

************
Toy example
************
By default, a Component must complete before the next one runs. We can enable one
component to start in parallel which allows the code to proceed without having
to wait for the first one to finish.

.. lit_tabs::
   :descriptions: No parallel components; Allow the train component to run in parallel; When the component runs, it will run in parallel; The next component is unblocked and can now immediately run.
   :code_files: /workflows/scripts/parallel/toy_app.py; /workflows/scripts/parallel/toy_parallel.py; /workflows/scripts/parallel/toy_parallel.py; /workflows/scripts/parallel/toy_parallel.py;
   :highlights: ; 18; 23; 24;
   :enable_run: true
   :tab_rows: 3
   :height: 540px

----

*******************************
Multiple components in parallel
*******************************
In this example, we start all 3 components at once. The first two start in parallel, which
allows the third component to run without waiting for the others to finish.

.. lit_tabs::
   :descriptions: No parallel components; Enable 2 components to run in parallel; Start both components together in parallel; Last component is not blocked and can start immediately.
   :code_files: /workflows/scripts/parallel/toy_two_parallel_not_started.py; /workflows/scripts/parallel/toy_two_parallel.py; /workflows/scripts/parallel/toy_two_parallel.py; /workflows/scripts/parallel/toy_two_parallel.py
   :highlights: ; 18, 19; 23, 24; 25
   :enable_run: true
   :tab_rows: 3
   :height: 540px
