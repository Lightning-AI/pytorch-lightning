#################
Conversational AI
#################

These are amazing ecosystems to help with Automatic Speech Recognition (ASR), Natural Language Processing (NLP), and Text to speech (TTS).

----

****
NeMo
****

`NVIDIA NeMo <https://github.com/NVIDIA/NeMo>`_ is a toolkit for building new State-of-the-Art
Conversational AI models. NeMo has separate collections for Automatic Speech Recognition (ASR),
Natural Language Processing (NLP), and Text-to-Speech (TTS) models. Each collection consists of
prebuilt modules that include everything needed to train on your data.
Every module can easily be customized, extended, and composed to create new Conversational AI
model architectures.

Conversational AI architectures are typically very large and require a lot of data  and compute
for training. NeMo uses PyTorch Lightning for easy and performant multi-GPU/multi-node
mixed-precision training.

.. note:: Every NeMo model is a LightningModule that comes equipped with all supporting infrastructure for training and reproducibility.

----------

NeMo Models
===========

NeMo Models contain everything needed to train and reproduce state of the art Conversational AI
research and applications, including:

- neural network architectures
- datasets/data loaders
- data preprocessing/postprocessing
- data augmentors
- optimizers and schedulers
- tokenizers
- language models

NeMo uses `Hydra <https://hydra.cc/>`_ for configuring both NeMo models and the PyTorch Lightning Trainer.
Depending on the domain and application, many different AI libraries will have to be configured
to build the application. Hydra makes it easy to bring all of these libraries together
so that each can be configured from .yaml or the Hydra CLI.

.. note:: Every NeMo model has an example configuration file and a corresponding script that contains all configurations needed for training.

The end result of using NeMo, Pytorch Lightning, and Hydra is that
NeMo models all have the same look and feel. This makes it easy to do Conversational AI research
across multiple domains. NeMo models are also fully compatible with the PyTorch ecosystem.

Installing NeMo
---------------

Before installing NeMo, please install Cython first.

.. code-block:: bash

    pip install Cython

For ASR and TTS models, also install these linux utilities.

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg

Then installing the latest NeMo release is a simple pip install.

.. code-block:: bash

    pip install nemo_toolkit[all]==1.0.0b1

To install the main branch from GitHub:

.. code-block:: bash

    python -m pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[all]

To install from a local clone of NeMo:

.. code-block:: bash

    ./reinstall.sh # from cloned NeMo's git root

For Docker users, the NeMo container is available on
`NGC <https://ngc.nvidia.com/catalog/containers/nvidia:nemo>`_.

.. code-block:: bash

    docker pull nvcr.io/nvidia/nemo:v1.0.0b1

.. code-block:: bash

    docker run --runtime=nvidia -it --rm -v --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/nemo:v1.0.0b1

Experiment Manager
------------------

NeMo's Experiment Manager leverages PyTorch Lightning for model checkpointing,
TensorBoard Logging, and Weights and Biases logging. The Experiment Manager is included by default
in all NeMo example scripts.

.. code-block:: python

    exp_manager(trainer, cfg.get("exp_manager", None))

And is configurable via .yaml with Hydra.

.. code-block:: bash

    exp_manager:
        exp_dir: null
        name: *name
        create_tensorboard_logger: True
        create_checkpoint_callback: True

Optionally launch Tensorboard to view training results in ./nemo_experiments (by default).

.. code-block:: bash

    tensorboard --bind_all --logdir nemo_experiments

--------

Automatic Speech Recognition (ASR)
==================================

Everything needed to train Convolutional ASR models is included with NeMo.
NeMo supports multiple Speech Recognition architectures, including Jasper and QuartzNet.
`NeMo Speech Models <https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels>`_
can be trained from scratch on custom datasets or
fine-tuned using pre-trained checkpoints trained on thousands of hours of audio
that can be restored for immediate use.

Some typical ASR tasks are included with NeMo:

- `Audio transcription <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/tutorials/asr/01_ASR_with_NeMo.ipynb>`_
- `Byte Pair/Word Piece Training <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/examples/asr/speech_to_text_bpe.py>`_
- `Speech Commands <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/tutorials/asr/03_Speech_Commands.ipynb>`_
- `Voice Activity Detection <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/tutorials/asr/06_Voice_Activiy_Detection.ipynb>`_
- `Speaker Recognition <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/examples/speaker_recognition/speaker_reco.py>`_

See this `asr notebook <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/tutorials/asr/01_ASR_with_NeMo.ipynb>`_
for a full tutorial on doing ASR with NeMo, PyTorch Lightning, and Hydra.

Specify ASR Model Configurations with YAML File
-----------------------------------------------

NeMo Models and the PyTorch Lightning Trainer can be fully configured from .yaml files using Hydra.

See this `asr config <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/examples/asr/conf/config.yaml>`_
for the entire speech to text .yaml file.

.. code-block:: yaml

    # configure the PyTorch Lightning Trainer
    trainer:
        gpus: 0 # number of gpus
        max_epochs: 5
        max_steps: null # computed at runtime if not set
        num_nodes: 1
        accelerator: ddp
        ...
    # configure the ASR model
    model:
        ...
        encoder:
            cls: nemo.collections.asr.modules.ConvASREncoder
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
        # all other configuration, data, optimizer, preprocessor, etc
        ...

Developing ASR Model From Scratch
---------------------------------

`speech_to_text.py <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/examples/asr/speech_to_text.py>`_

.. code-block:: python

    # hydra_runner calls hydra.main and is useful for multi-node experiments
    @hydra_runner(config_path="conf", config_name="config")
    def main(cfg):
        trainer = Trainer(**cfg.trainer)
        asr_model = EncDecCTCModel(cfg.model, trainer)
        trainer.fit(asr_model)


Hydra makes every aspect of the NeMo model,
including the PyTorch Lightning Trainer, customizable from the command line.

.. code-block:: bash

    python NeMo/examples/asr/speech_to_text.py --config-name=quartznet_15x5 \
        trainer.gpus=4 \
        trainer.max_epochs=128 \
        +trainer.precision=16 \
        model.train_ds.manifest_filepath=<PATH_TO_DATA>/librispeech-train-all.json \
        model.validation_ds.manifest_filepath=<PATH_TO_DATA>/librispeech-dev-other.json \
        model.train_ds.batch_size=64 \
        +model.validation_ds.num_workers=16 \
        +model.train_ds.num_workers=16

.. note:: Training NeMo ASR models can take days/weeks so it is highly recommended to use multiple GPUs and multiple nodes with the PyTorch Lightning Trainer.


Using State-Of-The-Art Pre-trained ASR Model
--------------------------------------------

Transcribe audio with QuartzNet model pretrained on ~3300 hours of audio.

.. code-block:: python

    quartznet = EncDecCTCModel.from_pretrained('QuartzNet15x5Base-En')

    files = ['path/to/my.wav'] # file duration should be less than 25 seconds

    for fname, transcription in zip(files, quartznet.transcribe(paths2audio_files=files)):
        print(f"Audio in {fname} was recognized as: {transcription}")

To see the available pretrained checkpoints:

.. code-block:: python

    EncDecCTCModel.list_available_models()

NeMo ASR Model Under the Hood
-----------------------------

Any aspect of ASR training or model architecture design can easily be customized
with PyTorch Lightning since every NeMo model is a Lightning Module.

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
            self.log_dict({
                'train_loss': loss_value,
                'training_batch_wer': wer_num / wer_denom,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
            })
            return loss_value

Neural Types in NeMo ASR
------------------------

NeMo Models and Neural Modules come with Neural Type checking.
Neural type checking is extremely useful when combining many different neural
network architectures for a production-grade application.

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

--------

Natural Language Processing (NLP)
=================================

Everything needed to finetune BERT-like language models for NLP tasks is included with NeMo.
`NeMo NLP Models <https://ngc.nvidia.com/catalog/models/nvidia:nemonlpmodels>`_
include `HuggingFace Transformers <https://github.com/huggingface/transformers>`_
and `NVIDIA Megatron-LM <https://github.com/NVIDIA/Megatron-LM>`_ BERT and Bio-Megatron models.
NeMo can also be used for pretraining BERT-based language models from HuggingFace.

Any of the HuggingFace encoders or Megatron-LM encoders can easily be used for the NLP tasks
that are included with NeMo:

- `Glue Benchmark (All tasks) <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/tutorials/nlp/GLUE_Benchmark.ipynb>`_
- `Intent Slot Classification <https://github.com/NVIDIA/NeMo/tree/v1.0.0b1/examples/nlp/intent_slot_classification>`_
- `Language Modeling (BERT Pretraining) <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/tutorials/nlp/01_Pretrained_Language_Models_for_Downstream_Tasks.ipynb>`_
- `Question Answering <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/tutorials/nlp/Question_Answering_Squad.ipynb>`_
- `Text Classification <https://github.com/NVIDIA/NeMo/tree/v1.0.0b1/examples/nlp/text_classification>`_ (including Sentiment Analysis)
- `Token Classification <https://github.com/NVIDIA/NeMo/tree/v1.0.0b1/examples/nlp/token_classification>`_ (including Named Entity Recognition)
- `Punctuation and Capitalization <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/tutorials/nlp/Punctuation_and_Capitalization.ipynb>`_

Named Entity Recognition (NER)
------------------------------

NER (or more generally token classification) is the NLP task of detecting and classifying key information (entities) in text.
This task is very popular in Healthcare and Finance. In finance, for example, it can be important to identify
geographical, geopolitical, organizational, persons, events, and natural phenomenon entities.
See this `NER notebook <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/tutorials/nlp/Token_Classification_Named_Entity_Recognition.ipynb>`_
for a full tutorial on doing NER with NeMo, PyTorch Lightning, and Hydra.

Specify NER Model Configurations with YAML File
-----------------------------------------------

.. note:: NeMo Models and the PyTorch Lightning Trainer can be fully configured from .yaml files using Hydra.

See this `token classification config <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/examples/nlp/token_classification/conf/token_classification_config.yaml>`_
for the entire NER (token classification) .yaml file.

.. code-block:: yaml

    # configure any argument of the PyTorch Lightning Trainer
    trainer:
        gpus: 1 # the number of gpus, 0 for CPU
        num_nodes: 1
        max_epochs: 5
        ...
    # configure any aspect of the token classification model here
    model:
        dataset:
            data_dir: ??? # /path/to/data
            class_balancing: null # choose from [null, weighted_loss]. Weighted_loss enables the weighted class balancing of the loss, may be used for handling unbalanced classes
            max_seq_length: 128
            ...
      tokenizer:
        tokenizer_name: ${model.language_model.pretrained_model_name} # or sentencepiece
        vocab_file: null # path to vocab file
        ...
    # the language model can be from HuggingFace or Megatron-LM
    language_model:
        pretrained_model_name: bert-base-uncased
        lm_checkpoint: null
        ...
    # the classifier for the downstream task
      head:
        num_fc_layers: 2
        fc_dropout: 0.5
        activation: 'relu'
        ...
    # all other configuration: train/val/test/ data, optimizer, experiment manager, etc
    ...

Developing NER Model From Scratch
---------------------------------

`token_classification.py <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/examples/nlp/token_classification/token_classification.py>`_

.. code-block:: python

    # hydra_runner calls hydra.main and is useful for multi-node experiments
    @hydra_runner(config_path="conf", config_name="token_classification_config")
    def main(cfg: DictConfig) -> None:
        trainer = pl.Trainer(**cfg.trainer)
        model = TokenClassificationModel(cfg.model, trainer=trainer)
        trainer.fit(model)

After training, we can do inference with the saved NER model using PyTorch Lightning.

Inference from file:

.. code-block:: python

    gpu = 1 if cfg.trainer.gpus != 0 else 0
    trainer = pl.Trainer(gpus=gpu)
    model.set_trainer(trainer)
    model.evaluate_from_file(
        text_file=os.path.join(cfg.model.dataset.data_dir, cfg.model.validation_ds.text_file),
        labels_file=os.path.join(cfg.model.dataset.data_dir, cfg.model.validation_ds.labels_file),
        output_dir=exp_dir,
        add_confusion_matrix=True,
        normalize_confusion_matrix=True,
    )

Or we can run inference on a few examples:

.. code-block:: python

    queries = ['we bought four shirts from the nvidia gear store in santa clara.', 'Nvidia is a company in Santa Clara.']
    results = model.add_predictions(queries)

    for query, result in zip(queries, results):
        logging.info(f'Query : {query}')
        logging.info(f'Result: {result.strip()}\n')

Hydra makes every aspect of the NeMo model, including the PyTorch Lightning Trainer, customizable from the command line.

.. code-block:: bash

    python token_classification.py \
        model.language_model.pretrained_model_name=bert-base-cased \
        model.head.num_fc_layers=2 \
        model.dataset.data_dir=/path/to/my/data  \
        trainer.max_epochs=5 \
        trainer.gpus=[0,1]

-----------

Tokenizers
----------

Tokenization is the process of converting natural language text into integer arrays
which can be used for machine learning.
For NLP tasks, tokenization is an essential part of data preprocessing.
NeMo supports all BERT-like model tokenizers from
`HuggingFace's AutoTokenizer <https://huggingface.co/transformers/model_doc/auto.html#autotokenizer>`_
and also supports `Google's SentencePieceTokenizer <https://github.com/google/sentencepiece>`_
which can be trained on custom data.

To see the list of supported tokenizers:

.. code-block:: python

    from nemo.collections import nlp as nemo_nlp

    nemo_nlp.modules.get_tokenizer_list()

See this `tokenizer notebook <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/tutorials/nlp/02_NLP_Tokenizers.ipynb>`_
for a full tutorial on using tokenizers in NeMo.

Language Models
---------------

Language models are used to extract information from (tokenized) text.
Much of the state-of-the-art in natural language processing is achieved
by fine-tuning pretrained language models on the downstream task.

With NeMo, you can either `pretrain <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/examples/nlp/language_modeling/bert_pretraining.py>`_
a BERT model on your data or use a pretrained language model from `HuggingFace Transformers <https://github.com/huggingface/transformers>`_
or `NVIDIA Megatron-LM <https://github.com/NVIDIA/Megatron-LM>`_.

To see the list of language models available in NeMo:

.. code-block:: python

    nemo_nlp.modules.get_pretrained_lm_models_list(include_external=True)

Easily switch between any language model in the above list by using `.get_lm_model`.

.. code-block:: python

    nemo_nlp.modules.get_lm_model(pretrained_model_name='distilbert-base-uncased')

See this `language model notebook <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/tutorials/nlp/01_Pretrained_Language_Models_for_Downstream_Tasks.ipynb>`_
for a full tutorial on using pretrained language models in NeMo.

Using a Pre-trained NER Model
-----------------------------

NeMo has pre-trained NER models that can be used
to get started with Token Classification right away.
Models are automatically downloaded from NGC,
cached locally to disk,
and loaded into GPU memory using the `.from_pretrained` method.

.. code-block:: python

    # load pre-trained NER model
    pretrained_ner_model = TokenClassificationModel.from_pretrained(model_name="NERModel")

    # define the list of queries for inference
    queries = [
        'we bought four shirts from the nvidia gear store in santa clara.',
        'Nvidia is a company.',
        'The Adventures of Tom Sawyer by Mark Twain is an 1876 novel about a young boy growing '
        + 'up along the Mississippi River.',
    ]
    results = pretrained_ner_model.add_predictions(queries)

    for query, result in zip(queries, results):
        print()
        print(f'Query : {query}')
        print(f'Result: {result.strip()}\n')

NeMo NER Model Under the Hood
-----------------------------

Any aspect of NLP training or model architecture design can easily be customized with PyTorch Lightning
since every NeMo model is a Lightning Module.

.. code-block:: python

    class TokenClassificationModel(ModelPT):
        """
        Token Classification Model with BERT, applicable for tasks such as Named Entity Recognition
        """
        ...
        @typecheck()
        def forward(self, input_ids, token_type_ids, attention_mask):
            hidden_states = self.bert_model(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )
            logits = self.classifier(hidden_states=hidden_states)
            return logits

        # PTL-specfic methods
        def training_step(self, batch, batch_idx):
            """
            Lightning calls this inside the training loop with the data from the training dataloader
            passed in as `batch`.
            """
            input_ids, input_type_ids, input_mask, subtokens_mask, loss_mask, labels = batch
            logits = self(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

            loss = self.loss(logits=logits, labels=labels, loss_mask=loss_mask)
            self.log_dict({'train_loss': loss, 'lr': self._optimizer.param_groups[0]['lr']})
            return loss
        ...

Neural Types in NeMo NLP
------------------------

NeMo Models and Neural Modules come with Neural Type checking.
Neural type checking is extremely useful when combining many different neural network architectures
for a production-grade application.

.. code-block:: python

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.classifier.output_types

--------

Text-To-Speech (TTS)
====================

Everything needed to train TTS models and generate audio is included with NeMo.
`NeMo TTS Models <https://ngc.nvidia.com/catalog/models/nvidia:nemottsmodels>`_
can be trained from scratch on your own data or pretrained models can be downloaded
automatically. NeMo currently supports  a two step inference procedure.
First, a model is used to generate a mel spectrogram from text.
Second, a model is used to generate audio from a mel spectrogram.

Mel Spectrogram Generators:

- `Tacotron 2 <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/examples/tts/tacotron2.py>`_
- `Glow-TTS <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/examples/tts/glow_tts.py>`_

Audio Generators:

- Griffin-Lim
- `WaveGlow <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/examples/tts/waveglow.py>`_
- `SqueezeWave <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/examples/tts/squeezewave.py>`_


Specify TTS Model Configurations with YAML File
-----------------------------------------------

.. note:: NeMo Models and PyTorch Lightning Trainer can be fully configured from .yaml files using Hydra.

`tts/conf/glow_tts.yaml <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/examples/tts/conf/glow_tts.yaml>`_

.. code-block:: yaml

    # configure the PyTorch Lightning Trainer
    trainer:
        gpus: -1 # number of gpus
        max_epochs: 350
        num_nodes: 1
        accelerator: ddp
        ...

    # configure the TTS model
    model:
        ...
        encoder:
            cls: nemo.collections.tts.modules.glow_tts.TextEncoder
                params:
                n_vocab: 148
                out_channels: *n_mels
                hidden_channels: 192
                filter_channels: 768
                filter_channels_dp: 256
                ...
    # all other configuration, data, optimizer, parser, preprocessor, etc
    ...

Developing TTS Model From Scratch
---------------------------------

`tts/glow_tts.py <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/examples/tts/glow_tts.py>`_

.. code-block:: python

    # hydra_runner calls hydra.main and is useful for multi-node experiments
    @hydra_runner(config_path="conf", config_name="glow_tts")
    def main(cfg):
        trainer = pl.Trainer(**cfg.trainer)
        model = GlowTTSModel(cfg=cfg.model, trainer=trainer)
        trainer.fit(model)

Hydra makes every aspect of the NeMo model, including the PyTorch Lightning Trainer, customizable from the command line.

.. code-block:: bash

    python NeMo/examples/tts/glow_tts.py \
        trainer.gpus=4 \
        trainer.max_epochs=400 \
        ...
        train_dataset=/path/to/train/data \
        validation_datasets=/path/to/val/data \
        model.train_ds.batch_size = 64 \

.. note:: Training NeMo TTS models from scratch can take days or weeks so it is highly recommended to use multiple GPUs and multiple nodes with the PyTorch Lightning Trainer.

Using State-Of-The-Art Pre-trained TTS Model
--------------------------------------------

Generate speech using models trained on `LJSpeech <https://keithito.com/LJ-Speech-Dataset/>`,
around 24 hours of single speaker data.

See this `TTS notebook <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/tutorials/tts/1_TTS_inference.ipynb>`_
for a full tutorial on generating speech with NeMo, PyTorch Lightning, and Hydra.

.. code-block:: python

    # load pretrained spectrogram model
    spec_gen = SpecModel.from_pretrained('GlowTTS-22050Hz').cuda()

    # load pretrained Generators
    vocoder = WaveGlowModel.from_pretrained('WaveGlow-22050Hz').cuda()

    def infer(spec_gen_model, vocder_model, str_input):
        with torch.no_grad():
            parsed = spec_gen.parse(text_to_generate)
            spectrogram = spec_gen.generate_spectrogram(tokens=parsed)
            audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
        if isinstance(spectrogram, torch.Tensor):
            spectrogram = spectrogram.to('cpu').numpy()
        if len(spectrogram.shape) == 3:
            spectrogram = spectrogram[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.to('cpu').numpy()
        return spectrogram, audio

    text_to_generate = input("Input what you want the model to say: ")
    spec, audio = infer(spec_gen, vocoder, text_to_generate)

To see the available pretrained checkpoints:

.. code-block:: python

    # spec generator
    GlowTTSModel.list_available_models()

    # vocoder
    WaveGlowModel.list_available_models()

NeMo TTS Model Under the Hood
-----------------------------

Any aspect of TTS training or model architecture design can easily
be customized with PyTorch Lightning since every NeMo model is a LightningModule.

`glow_tts.py <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/nemo/collections/tts/models/glow_tts.py>`_

.. code-block:: python

    class GlowTTSModel(SpectrogramGenerator):
        """
        GlowTTS model used to generate spectrograms from text
        Consists of a text encoder and an invertible spectrogram decoder
        """
        ...
        # NeMo models come with neural type checking
        @typecheck(
            input_types={
                "x": NeuralType(('B', 'T'), TokenIndex()),
                "x_lengths": NeuralType(('B'), LengthsType()),
                "y": NeuralType(('B', 'D', 'T'), MelSpectrogramType(), optional=True),
                "y_lengths": NeuralType(('B'), LengthsType(), optional=True),
                "gen": NeuralType(optional=True),
                "noise_scale": NeuralType(optional=True),
                "length_scale": NeuralType(optional=True),
            }
        )
        def forward(self, *, x, x_lengths, y=None, y_lengths=None, gen=False, noise_scale=0.3, length_scale=1.0):
            if gen:
                return self.glow_tts.generate_spect(
                    text=x, text_lengths=x_lengths, noise_scale=noise_scale, length_scale=length_scale
                )
            else:
                return self.glow_tts(text=x, text_lengths=x_lengths, spect=y, spect_lengths=y_lengths)
        ...
        def step(self, y, y_lengths, x, x_lengths):
            z, y_m, y_logs, logdet, logw, logw_, y_lengths, attn = self(
                x=x, x_lengths=x_lengths, y=y, y_lengths=y_lengths, gen=False
            )

            l_mle, l_length, logdet = self.loss(
                z=z,
                y_m=y_m,
                y_logs=y_logs,
                logdet=logdet,
                logw=logw,
                logw_=logw_,
                x_lengths=x_lengths,
                y_lengths=y_lengths,
            )

            loss = sum([l_mle, l_length])

            return l_mle, l_length, logdet, loss, attn

        # PTL-specfic methods
        def training_step(self, batch, batch_idx):
            y, y_lengths, x, x_lengths = batch

            y, y_lengths = self.preprocessor(input_signal=y, length=y_lengths)

            l_mle, l_length, logdet, loss, _ = self.step(y, y_lengths, x, x_lengths)

            self.log_dict({"l_mle": l_mle, "l_length": l_length, "logdet": logdet}, prog_bar=True)
            return loss
        ...

Neural Types in NeMo TTS
------------------------

NeMo Models and Neural Modules come with Neural Type checking.
Neural type checking is extremely useful when combining many different neural network architectures
for a production-grade application.

.. code-block:: python

    @typecheck(
        input_types={
            "x": NeuralType(('B', 'T'), TokenIndex()),
            "x_lengths": NeuralType(('B'), LengthsType()),
            "y": NeuralType(('B', 'D', 'T'), MelSpectrogramType(), optional=True),
            "y_lengths": NeuralType(('B'), LengthsType(), optional=True),
            "gen": NeuralType(optional=True),
            "noise_scale": NeuralType(optional=True),
            "length_scale": NeuralType(optional=True),
        }
    )
    def forward(self, *, x, x_lengths, y=None, y_lengths=None, gen=False, noise_scale=0.3, length_scale=1.0):
        ...

--------

Learn More
==========

- Watch the `NVIDIA NeMo Intro Video <https://youtu.be/wBgpMf_KQVw>`_
- Watch the `PyTorch Lightning and NVIDIA NeMo Discussion Video <https://youtu.be/rFAX1-4DSr4>`_
- Visit the `NVIDIA NeMo Developer Website <https://developer.nvidia.com/nvidia-nemo>`_
- Read the `NVIDIA NeMo PyTorch Blog <https://medium.com/pytorch/nvidia-nemo-neural-modules-and-models-for-conversational-ai-d660480d9696>`_
- Download pre-trained `ASR <https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels>`_, `NLP <https://ngc.nvidia.com/catalog/models/nvidia:nemonlpmodels>`_, and `TTS <https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels>`_ models on `NVIDIA NGC <https://ngc.nvidia.com/>`_ to quickly get started with NeMo.
- Become an expert on Building Conversational AI applications with our `tutorials <https://github.com/NVIDIA/NeMo#tutorials>`_, and `example scripts <https://github.com/NVIDIA/NeMo/tree/v1.0.0b1/examples>`_,
- See our `developer guide <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/>`_ for more information on core NeMo concepts, ASR/NLP/TTS collections, and the NeMo API.

.. note:: NeMo tutorial notebooks can be run on `Google Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_.

NVIDIA `NeMo <https://github.com/NVIDIA/NeMo>`_ is actively being developed on GitHub.
`Contributions <https://github.com/NVIDIA/NeMo/blob/v1.0.0b1/CONTRIBUTING.md>`_ are welcome!
