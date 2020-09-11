Conversational AI
-----------------

Using NeMo Models
^^^^^^^^^^^^^^^^^

NeMo is a toolkit for doing research in Conversational AI.   NeMo makes it easy to build complex 
automatic speech recognition (ASR), natural language processing (NLP), and text-to-speech (TTS) 
applications.

Conversational AI architectures are typically very large and require a lot of data  and compute 
for training. NeMo uses PyTorch Lightning for for easy and performant multi-gpu/multi-node 
mixed-precision training. 

.. note:: Every NeMo model is a LightningModule that comes equipped with all supporting infrastructure for training and reproducibility.

NeMo Models contain everything needed to to train and reproduce state of the art Conversational AI
research and applications. This includes

- neural network architectures 
- datasets/dataloaders
- data preprocessing/postprocessing
- data augmentors
- optimizers and schedulers
- tokenizers, language models

NeMo uses Hydra for configuring both NeMo models and the PyTorch Lightning Trainer.
Depending on the domain and application, many different AI libraries will have to be configured
to build the application. Hydra makes it easy to bring all of these libraries together
and do all the configuration from .yaml or the Hydra CLI.

.. note:: Every NeMo model has an example configuration and run script that contains all configuration needed for training.

Installing NeMo
^^^^^^^^^^^^^^^

Installing the latest NeMo release is a simple pip install.

.. code-block:: bash

    pip install nemo_toolkit[all]

To install a specific branch from GitHub:

.. code-block:: bash

    python -m pip install git+https://github.com/NVIDIA/NeMo.git@{BRANCH}#egg=nemo_toolkit[nlp]

.. note:: Replace {BRANCH} with the specific branch name from GitHub.

Example: Speech to Text (ASR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train Convolutional ASR models with NeMo and PyTorch Lightning.

.. code-block:: python

    trainer = Trainer(**cfg.trainer)
    asr_model = EncDecCTCModel(cfg.model, trainer)
    trainer.fit(asr_model)

.. note:: NeMo models and PyTorch Lightning Trainer can be fully configured from .yaml files using Hydra. 

Training NeMo models with PyTorch Lightning and Hydra is simple from the command line.

.. code-block:: bash

    python NeMo/examples/asr/speech_to_text.py --config-name=quartznet_15x5 \
        trainer.gpus=4 \
        trainer.max_epochs=128 \
        +trainer.precision=16 \
        +trainer.amp_level=O1 \
        model.train_ds.manifest_filepath=<PATH_TO_DATA>/librispeech-train-all.json \
        model.validation_ds.manifest_filepath=<PATH_TO_DATA>/librispeech-dev-other.json \
        model.train_ds.batch_size=64 \
        +model.validation_ds.num_workers=16 \
        +model.train_ds.num_workers=16

.. note:: Training NeMo ASR models can take days/weeks so it is highly recommended to use multiple GPUs and multiple nodes with the PyTorch Lightning Trainer.

Optionally launch Tensorboard to view training results

.. code-block:: bash

    tensorboard --bind_all --logdir nemo_experiments


Transcribe audio with QuartzNet pretrained on 7000+ hours of audio.

.. code-block:: python

    quartznet = EncDecCTCModel.from_pretrained('QuartzNet15x5Base-En')

    files = ['path/to/my.wav'] # file should be less than 25 seconds

    for fname, transcription in zip(files, quartznet.transcribe(paths2audio_files=files)):
        print(f"Audio in {fname} was recognized as: {transcription}")

Example: Voice Activity Detection (VAD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train a MatchboxNet model with a modified decoder head for recognizing speakers.

.. code-block:: python

    trainer = Trainer(**cfg.trainer)
    speaker_model = EncDecSpeakerLabelModel(cfg=cfg.model, trainer=trainer)
    trainer.fit(speaker_model)