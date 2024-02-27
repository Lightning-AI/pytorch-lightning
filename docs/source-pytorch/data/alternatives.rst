:orphan:

.. _dataiters:

Using 3rd Party Data Iterables
==============================

When training a model on a specific task, data loading and preprocessing might become a bottleneck.
Lightning does not enforce a specific data loading approach nor does it try to control it.
The only assumption Lightning makes is that a valid iterable is provided.

For PyTorch-based programs, these iterables are typically instances of :class:`~torch.utils.data.DataLoader`.
However, Lightning also supports other data types such as a list of batches, generators, or other custom iterables or
collections of the former.

.. code-block:: python

    # random list of batches
    data = [(torch.rand(32, 3, 32, 32), torch.randint(0, 10, (32,))) for _ in range(100)]
    model = LitClassifier()
    trainer = Trainer()
    trainer.fit(model, data)

Below we showcase Lightning examples with packages that compete with the generic PyTorch DataLoader and might be
faster depending on your use case. They might require custom data serialization, loading, and preprocessing that
is often hardware accelerated.

StreamingDataset
^^^^^^^^^^^^^^^^

As datasets grow in size and the number of nodes scales, loading training data can become a significant challenge.
The `StreamingDataset <https://github.com/mosaicml/streaming>`__ can make training on large datasets from cloud storage
as fast, cheap, and scalable as possible.

This library uses a custom built :class:`~torch.utils.data.IterableDataset`. The library recommends iterating through it
via a regular :class:`~torch.utils.data.DataLoader`. This means that support in the ``Trainer`` is seamless:

.. code-block:: python

    import lightning as L
    from streaming import MDSWriter, StreamingDataset


    class YourDataset(StreamingDataset):
        ...


    # you could do this in the `prepare_data` hook too
    with MDSWriter(out="...", columns=...) as out:
        out.write(...)

    train_dataset = YourDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    model = ...
    trainer = L.Trainer()
    trainer.fit(model, train_dataloader)

FFCV
^^^^

Taking the example from the `FFCV <https://github.com/libffcv/ffcv>`__ readme, we can use it with Lightning
by just removing the hardcoded ``ToDevice(0)`` as Lightning takes care of GPU placement. In case you want to use some
data transformations on GPUs, change the ``ToDevice(0)`` to ``ToDevice(self.trainer.local_rank)`` to correctly map to
the desired GPU in your pipeline. When moving data to a specific device, you can always refer to
``self.trainer.local_rank`` to get the accelerator used by the current process.

.. code-block:: python

    import lightning as L
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Cutout
    from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder

    # Random resized crop
    decoder = RandomResizedCropRGBImageDecoder((224, 224))
    # Data decoding and augmentation
    image_pipeline = [decoder, Cutout(), ToTensor(), ToTorchImage()]
    label_pipeline = [IntDecoder(), ToTensor()]
    # Pipeline for each data field
    pipelines = {"image": image_pipeline, "label": label_pipeline}
    # Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
    train_dataloader = Loader(
        write_path, batch_size=bs, num_workers=num_workers, order=OrderOption.RANDOM, pipelines=pipelines
    )

    model = ...
    trainer = L.Trainer()
    trainer.fit(model, train_dataloader)

WebDataset
^^^^^^^^^^

The `WebDataset <https://webdataset.github.io/webdataset>`__ makes it easy to write I/O pipelines for large datasets.
Datasets can be stored locally or in the cloud. ``WebDataset`` is just an instance of a standard IterableDataset.
The webdataset library contains a small wrapper (``WebLoader``) that adds a fluid interface to the DataLoader (and is otherwise identical).

.. code-block:: python

    import lightning as L
    import webdataset as wds

    dataset = wds.WebDataset(urls)
    train_dataloader = wds.WebLoader(dataset)

    model = ...
    trainer = L.Trainer()
    trainer.fit(model, train_dataloader)

You can find a complete example `here <https://github.com/webdataset/webdataset-lightning>`__.

NVIDIA DALI
^^^^^^^^^^^

By just changing ``device_id=0`` to ``device_id=self.trainer.local_rank`` we can also leverage DALI's GPU decoding:

.. code-block:: python

    import lightning as L
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    import os

    # To run with different data, see documentation of nvidia.dali.fn.readers.file
    # points to https://github.com/NVIDIA/DALI_extra
    data_root_dir = os.environ["DALI_EXTRA_PATH"]
    images_dir = os.path.join(data_root_dir, "db", "single", "jpeg")


    @pipeline_def(num_threads=4, device_id=self.trainer.local_rank)
    def get_dali_pipeline():
        images, labels = fn.readers.file(file_root=images_dir, random_shuffle=True, name="Reader")
        # decode data on the GPU
        images = fn.decoders.image_random_crop(images, device="mixed", output_type=types.RGB)
        # the rest of processing happens on the GPU as well
        images = fn.resize(images, resize_x=256, resize_y=256)
        images = fn.crop_mirror_normalize(
            images,
            crop_h=224,
            crop_w=224,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=fn.random.coin_flip(),
        )
        return images, labels


    train_dataloader = DALIGenericIterator(
        [get_dali_pipeline(batch_size=16)],
        ["data", "label"],
        reader_name="Reader",
    )

    model = ...
    trainer = L.Trainer()
    trainer.fit(model, train_dataloader)

You can find a complete tutorial `here <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/frameworks/pytorch/pytorch-lightning.html>`__.


Limitations
------------
Lightning works with all kinds of custom data iterables as shown above. There are, however, a few features that cannot
be supported this way. These restrictions come from the fact that for their support,
Lightning needs to know a lot on the internals of these iterables.

- In a distributed multi-GPU setting (ddp), Lightning wraps the DataLoader's sampler with a wrapper for distributed
  support. This makes sure that each GPU sees a different part of the dataset. As sampling can be implemented in
  arbitrary ways with custom iterables, Lightning might not be able to do this for you. If this is the case, you can use
  the :paramref:`~lightning.pytorch.trainer.trainer.Trainer.use_distributed_sampler` argument to disable this logic and
  set the distributed sampler yourself.
