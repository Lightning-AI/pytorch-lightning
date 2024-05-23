
.. toctree::
   :maxdepth: 1
   :hidden:

   2D Parallelism <../advanced/model_parallel/tp_fsdp>
   Accelerators <../extensions/accelerator>
   Callback <../extensions/callbacks>
   Checkpointing <../common/checkpointing>
   Cluster <../clouds/cluster>
   Cloud checkpoint <../common/checkpointing_advanced>
   Compile <../advanced/compile>
   Console Logging <../common/console_logs>
   Debugging <../debug/debugging>
   DeepSpeed <../advanced/model_parallel/deepspeed>
   Distributed Checkpoints <../common/checkpointing_expert>
   Early stopping <../common/early_stopping>
   Experiment manager (Logger) <../visualize/experiment_managers>
   Finetuning <../advanced/finetuning>
   FSDP <../advanced/model_parallel/fsdp>
   GPU <../accelerators/gpu>
   Half precision <../common/precision>
   HPU <../integrations/hpu/index>
   Inference <../deploy/production_intermediate>
   Lightning CLI <../cli/lightning_cli>
   LightningDataModule <../data/datamodule>
   LightningModule <../common/lightning_module>
   Log <../visualize/loggers>
   TPU <../accelerators/tpu>
   Metrics <https://torchmetrics.readthedocs.io/en/stable/>
   Model <../model/build_model.rst>
   Model Parallel <../advanced/model_parallel/index>
   Plugins <../extensions/plugins>
   Progress bar <../common/progress_bar>
   Production <../deploy/production_advanced>
   Predict <../deploy/production_basic>
   Pretrained models <../advanced/pretrained>
   Profiler <../tuning/profiler>
   Pruning and Quantization <../advanced/pruning_quantization>
   Remote filesystem and FSSPEC <../common/remote_fs>
   Strategy <../extensions/strategy>
   Strategy registry <../advanced/strategy_registry>
   Strategy integrations <../integrations/strategies/index>
   Style guide <../starter/style_guide>
   SWA <../advanced/training_tricks>
   SLURM <../clouds/cluster_advanced>
   Tensor Parallel <../advanced/model_parallel/tp>
   Transfer learning <../advanced/transfer_learning>
   Trainer <../common/trainer>
   TorchRun (TorchElastic) <../clouds/cluster_intermediate_2>
   Warnings <../advanced/warnings>


########
Glossary
########

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: 2D Parallelism
   :description: Combine Tensor Parallelism with FSDP (2D Parallel) to train efficiently on 100s of GPUs
   :col_css: col-md-12
   :button_link: ../advanced/model_parallel/tp_fsdp.html
   :height: 100

.. displayitem::
   :header: Accelerators
   :description: Accelerators connect the Trainer to hardware to train faster
   :col_css: col-md-12
   :button_link: ../extensions/accelerator.html
   :height: 100

.. displayitem::
   :header: Callback
   :description: Add self-contained extra functionality during training execution
   :col_css: col-md-12
   :button_link: ../extensions/callbacks.html
   :height: 100

.. displayitem::
   :header: Checkpointing
   :description: Save and load progress with checkpoints
   :col_css: col-md-12
   :button_link: ../common/checkpointing.html
   :height: 100

.. displayitem::
   :header: Cluster
   :description: Run on your own group of servers
   :col_css: col-md-12
   :button_link: ../clouds/cluster.html
   :height: 100

.. displayitem::
   :header: Cloud checkpoint
   :description: Save your models to cloud filesystems
   :col_css: col-md-12
   :button_link: ../common/checkpointing_advanced.html
   :height: 100

.. displayitem::
   :header: Compile
   :description: Use torch.compile to speed up models on modern hardware
   :col_css: col-md-12
   :button_link: ../advanced/compile.html
   :height: 100

.. displayitem::
   :header: Console Logging
   :description: Capture more visible logs
   :col_css: col-md-12
   :button_link: ../common/console_logs.html
   :height: 100

.. displayitem::
   :header: Debugging
   :description: Fix errors in your code
   :col_css: col-md-12
   :button_link: ../debug/debugging.html
   :height: 100

.. displayitem::
   :header: DeepSpeed
   :description: Distribute models with billions of parameters across hundreds GPUs
   :col_css: col-md-12
   :button_link: ../advanced/model_parallel/deepspeed.html
   :height: 100

.. displayitem::
   :header: Distributed Checkpoints
   :description: Save and load very large models efficiently with distributed checkpoints
   :col_css: col-md-12
   :button_link: ../common/checkpointing_expert.html
   :height: 100

.. displayitem::
   :header: Early stopping
   :description: Stop the training when no improvement is observed
   :col_css: col-md-12
   :button_link: ../common/early_stopping.html
   :height: 100

.. displayitem::
   :header: Experiment manager (Logger)
   :description: Tools for tracking and visualizing artifacts and logs
   :col_css: col-md-12
   :button_link: ../visualize/experiment_managers.html
   :height: 100

.. displayitem::
   :header: Finetuning
   :description: Technique for training pretrained models
   :col_css: col-md-12
   :button_link: ../advanced/finetuning.html
   :height: 100

.. displayitem::
   :header: FSDP
   :description: Distribute models with billions of parameters across hundreds GPUs
   :col_css: col-md-12
   :button_link: ../advanced/model_parallel/fsdp.html
   :height: 100

.. displayitem::
   :header: GPU
   :description: Graphics Processing Unit for faster training
   :col_css: col-md-12
   :button_link: ../accelerators/gpu.html
   :height: 100

.. displayitem::
   :header: Half precision
   :description: Using different numerical formats to save memory and run faster
   :col_css: col-md-12
   :button_link: ../common/precision.html
   :height: 100

.. displayitem::
   :header: HPU
   :description: Habana Gaudi AI Processor Unit for faster training
   :col_css: col-md-12
   :button_link: ../integrations/hpu/index.html
   :height: 100

.. displayitem::
   :header: Inference
   :description: Making predictions by applying a trained model to unlabeled examples
   :col_css: col-md-12
   :button_link: ../deploy/production_intermediate.html
   :height: 100

.. displayitem::
   :header: Lightning CLI
   :description: A Command-line Interface (CLI) to interact with Lightning code via a terminal
   :col_css: col-md-12
   :button_link: ../cli/lightning_cli.html
   :height: 100

.. displayitem::
   :header: LightningDataModule
   :description: A shareable, reusable class that encapsulates all the steps needed to process data
   :col_css: col-md-12
   :button_link: ../data/datamodule.html
   :height: 100

.. displayitem::
   :header: LightningModule
   :description: A base class organizug your neural network module
   :col_css: col-md-12
   :button_link: ../common/lightning_module.html
   :height: 100

.. displayitem::
   :header: Log
   :description: Outputs or results used for visualization and tracking
   :col_css: col-md-12
   :button_link: ../visualize/loggers.html
   :height: 100

.. displayitem::
   :header: Metrics
   :description: A statistic used to measure performance or other objectives we want to optimize
   :col_css: col-md-12
   :button_link: https://torchmetrics.readthedocs.io/en/stable/
   :height: 100

.. displayitem::
   :header: Model
   :description: The set of parameters and structure for a system to make predictions
   :col_css: col-md-12
   :button_link: ../model/build_model.html
   :height: 100

.. displayitem::
   :header: Model Parallelism
   :description: A way to scale training that splits a model between multiple devices.
   :col_css: col-md-12
   :button_link: ../advanced/model_parallel/index.html
   :height: 100

.. displayitem::
   :header: Plugins
   :description: Custom trainer integrations such as custom precision, checkpointing or cluster environment implementation
   :col_css: col-md-12
   :button_link: ../extensions/plugins.html
   :height: 100

.. displayitem::
   :header: Progress bar
   :description: Output printed to the terminal to visualize the progression of training
   :col_css: col-md-12
   :button_link: ../common/progress_bar.html
   :height: 100

.. displayitem::
   :header: Production
   :description: Using ML models in real world systems
   :col_css: col-md-12
   :button_link: ../deploy/production_advanced.html
   :height: 100

.. displayitem::
   :header: Prediction
   :description: Computing a model's output
   :col_css: col-md-12
   :button_link: ../deploy/production_basic.html
   :height: 100

.. displayitem::
   :header: Pretrained models
   :description: Models that have already been trained for a particular task
   :col_css: col-md-12
   :button_link: ../advanced/pretrained.html
   :height: 100

.. displayitem::
   :header: Profiler
   :description: Tool to identify bottlenecks and performance of different parts of a model
   :col_css: col-md-12
   :button_link: ../tuning/profiler.html
   :height: 100

.. displayitem::
   :header: Pruning
   :description: A technique to eliminae some of the model weights to reduce the model size and decrease inference requirements
   :col_css: col-md-12
   :button_link: ../advanced/pruning_quantization.html
   :height: 100

.. displayitem::
   :header: Quantization
   :description: A technique to accelerate the model inference speed and decrease the memory load while still maintaining the model accuracy
   :col_css: col-md-12
   :button_link: ../advanced/post_training_quantization.html
   :height: 100

.. displayitem::
   :header: Remote filesystem and FSSPEC
   :description: Accessing files from cloud storage providers
   :col_css: col-md-12
   :button_link: ../common/remote_fs.html
   :height: 100

.. displayitem::
   :header: Strategy
   :description: Ways the trainer controls the model distribution across training, evaluation, and prediction
   :col_css: col-md-12
   :button_link: ../extensions/strategy.html
   :height: 100

.. displayitem::
   :header: Strategy registry
   :description: A class that holds information about training strategies and allows adding new custom strategies
   :col_css: col-md-12
   :button_link: ../advanced/strategy_registry.html
   :height: 100

.. displayitem::
   :header: Style guide
   :description: Best practices to improve readability and reproducibility
   :col_css: col-md-12
   :button_link: ../starter/style_guide.html
   :height: 100

.. displayitem::
   :header: SWA
   :description: Stochastic Weight Averaging (SWA) can make your models generalize better
   :col_css: col-md-12
   :button_link: ../advanced/training_tricks.html#stochastic-weight-averaging
   :height: 100

.. displayitem::
   :header: SLURM
   :description: Simple Linux Utility for Resource Management, or simply Slurm, is a free and open-source job scheduler for Linux clusters
   :col_css: col-md-12
   :button_link: ../clouds/cluster_advanced.html
   :height: 100

.. displayitem::
   :header: Tensor Parallelism
   :description: Parallelize the computation of model layers across multiple GPUs, reducing memory usage and communication overhead
   :col_css: col-md-12
   :button_link: ../advanced/tp.html
   :height: 100

.. displayitem::
   :header: Transfer learning
   :description: Using pre-trained models to improve learning
   :col_css: col-md-12
   :button_link: ../advanced/transfer_learning.html
   :height: 100

.. displayitem::
   :header: Trainer
   :description: The class that automates and customizes model training
   :col_css: col-md-12
   :button_link: ../common/trainer.html
   :height: 100

.. displayitem::
   :header: Torch distributed
   :description: Setup for running on distributed environments
   :col_css: col-md-12
   :button_link: ../clouds/cluster_intermediate_2.html
   :height: 100

.. displayitem::
   :header: Warnings
   :description: Disable false-positive warnings emitted by Lightning
   :col_css: col-md-12
   :button_link: ../advanced/warnings.html
   :height: 100

.. raw:: html

        </div>
    </div>
