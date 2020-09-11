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
- tokenizers
- language models

NeMo uses Hydra for configuring both NeMo models and the PyTorch Lightning Trainer.
Depending on the domain and application, many different AI libraries will have to be configured
to build the application. Hydra makes it easy to bring all of these libraries together
and do all the configuration from .yaml or the Hydra CLI.

.. note:: Every NeMo model has an example configuration and run script that contains all configuration needed for training.

The end result of using NeMo, Pytorch Lightning, and Hydra is that
NeMo models all have the same look and feel so that is easy to do Conversational AI research
across multiple domains and all NeMo models are fully compatible with the PyTorch ecosystem.

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

Everything needed to train Convolutional ASR models is included with NeMo. 
NeMo supports multiple Speech Recognition architectures, including Jasper 
and QuartzNet. These models can be trained from scratch on custom datasets or 
pretrained checkpoints trained on thousands of hour of audio can be restored for
immediate use.

Some typical ASR tasks are included with NeMo:

- Audio transcription
- Speech Commands
- Voice Activity Detection
- Byte Pair/Word Piece Training

Configurations are in .yaml files included with NeMo/examples

.. code-block:: yaml

    # configure the PyTorch Lightning Trainer
    trainer:
        gpus: 0 # number of gpus
        max_epochs: 5
        max_steps: null # computed at runtime if not set
        num_nodes: 1
        distributed_backend: ddp
        ...
    # configure the ASR model
    encoder:
        _target_: nemo.collections.asr.modules.ConvASREncoder
        params:
        feat_in: *n_mels
        activation: relu
        conv_mask: true

        jasper:
            - filters: 128
            repeat: 1
            kernel: [11]
            stride: [1]
            dilation: [1]
            dropout: *dropout
            ...
    # all other configuration, data, optimizer, etc
    ...

The example speech-to-text script is just:

.. code-block:: python

    @hydra.main(config_name="config")
    def main(cfg):
        trainer = Trainer(**cfg.trainer)
        asr_model = EncDecCTCModel(cfg.model, trainer)
        trainer.fit(asr_model)

.. note:: NeMo models and PyTorch Lightning Trainer can be fully configured from .yaml files using Hydra. 

Hydra makes it so that every aspect of the NeMo model, 
including the PyTorch Lightning Trainer can be modified from the command line.

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

Any aspect of ASR training or model architecture design can easily be customized
since every NeMo model is a Lightning Module.

.. code-block:: python

    class EncDecCTCModel(ASRModel):
        """Base class for encoder decoder CTC-based models."""
    ...
        @typecheck()
        def forward(self, input_signal, input_signal_length):
            processed_signal, processed_signal_len = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )
            # Spec augment is not applied during evaluation/testing
            if self.spec_augmentation is not None and self.training:
                processed_signal = self.spec_augmentation(input_spec=processed_signal)
            encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_len)
            log_probs = self.decoder(encoder_output=encoded)
            greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
            return log_probs, encoded_len, greedy_predictions
    
        # PTL-specific methods
        def training_step(self, batch, batch_nb):
            audio_signal, audio_signal_len, transcript, transcript_len = batch
            log_probs, encoded_len, predictions = self.forward(
                input_signal=audio_signal, input_signal_length=audio_signal_len
            )
            loss_value = self.loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            wer_num, wer_denom = self._wer(predictions, transcript, transcript_len)
            tensorboard_logs = {
                'train_loss': loss_value,
                'training_batch_wer': wer_num / wer_denom,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
            }
            return {'loss': loss_value, 'log': tensorboard_logs}

Additionally, NeMo Models and Neural Modules come with Neural Type checking.
Neural type checking is extremely use when combining many different neural 
network architectures for a production grade application.

.. code-block:: python

        @property
        def input_types(self) -> Optional[Dict[str, NeuralType]]:
            if hasattr(self.preprocessor, '_sample_rate'):
                audio_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
            else:
                audio_eltype = AudioSignal()
            return {
                "input_signal": NeuralType(('B', 'T'), audio_eltype),
                "input_signal_length": NeuralType(tuple('B'), LengthsType()),
            }

        @property
        def output_types(self) -> Optional[Dict[str, NeuralType]]:
            return {
                "outputs": NeuralType(('B', 'T', 'D'), LogprobsType()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
                "greedy_predictions": NeuralType(('B', 'T'), LabelsType()),
            }


