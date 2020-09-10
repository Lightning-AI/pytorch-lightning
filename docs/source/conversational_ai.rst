Conversational AI
-----------------

Using NeMo Models
^^^^^^^^^^^^^^^^^

NeMo is a conversational AI toolkit that uses PyTorch Lightning for
training and fine-tuning of automatic speech recognition(ASR), 
natural language processing (NLP), and text-to-speech (TTS) applications and research.

.. note:: Every NeMo model is a LightningModule that comes equipped with all supporting infrastructure for training and reproducibility.

Example: Speech to Text (ASR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train Convolutional ASR models with NeMo and PyTorch Lightning.

.. code-block:: python

    trainer = Trainer(**cfg.trainer)
    asr_model = EncDecCTCModel(cfg.model, trainer)
    trainer.fit(asr_model)

.. note:: NeMo models and PyTorch Lightning Trainer can be fully configured from .yaml files using Hydra. 

Transcribe audio with QuartzNet pretrained on 1000's of hours of audio.

.. code-block:: python

    quartznet = EncDecCTCModel.from_pretrained('QuartzNet15x5Base-En')

    files = ['path/to/my.wav'] # file should be less than 25 seconds

    for fname, transcription in zip(files, quartznet.transcribe(paths2audio_files=files)):
        print(f"Audio in {fname} was recognized as: {transcription}")