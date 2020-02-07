import urllib.request
import zipfile
import os

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

url = 'https://s3.amazonaws.com/pytorch-tutorial-assets/imagenet_1k.zip'
filename = './imagenet_1k.zip'
dataset_root = './imagenet_1k'


def prepare_imagenet_1k():
    if not os.path.exists(dataset_root):
        if not os.path.exists(filename):
            urllib.request.urlretrieve(url, filename)

        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(".")


# modified from NVIDIA/DALI/docs/examples/pytorch/resnet50/main.py
class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, local_rank, world_size, data_dir, crop=224, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                                              seed=12 + local_rank)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        # let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers
        # to be able to handle all images from full-sized ImageNet
        # without additional re-allocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        img, labels = self.input(name="Reader")
        images = self.decode(img)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, local_rank, world_size, data_dir, crop=224, size=256):
        super(HybridValPipe, self).__init__(batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                                            seed=12 + local_rank)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        img, labels = self.input(name="Reader")
        images = self.decode(img)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, labels]
