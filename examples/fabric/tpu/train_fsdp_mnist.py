# Adapted from https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist_fsdp_with_ckpt.py

import argparse
from functools import partial

MODEL_OPTS = {
    '--flatten_parameters': {
        'action': 'store_true',
    },
    '--auto_wrap_policy': {
        'choices': ['none', 'size_based', 'type_based'],
        'default': 'none',
    },
    '--auto_wrap_min_num_params': {
        'type': int,
        'default': 1000,
    },
    '--use_nested_fsdp': {
        'action': 'store_false',
    },
    '--use_gradient_checkpointing': {
        'action': 'store_true',
    },
    '--ckpt_prefix': {
        'type': str,
        'default': '/tmp/mnist-fsdp/final_ckpt',
    },
    '--no_ckpt_consolidation': {
        'dest': 'ckpt_consolidation',
        'action': 'store_false',
    },
    '--compute_dtype': {
        'choices': ['float32', 'float16', 'bfloat16'],
        'default': 'float32',
    },
    '--fp32_reduce_scatter': {
        'action': 'store_true',
    },
    '--shard_param_on_dim_0': {
        'action': 'store_true',
    },
    '--no_pin_layout_in_collective_ops': {
        'action': 'store_false',
        'dest': 'pin_layout_in_collective_ops',
    },
}

import os
import shutil
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

import lightning as L
from lightning.fabric.strategies import XLAFSDPStrategy

from torch_xla.distributed.fsdp import (
    consolidate_sharded_model_checkpoints,
    checkpoint_module,
)


def parse_common_options(datadir=None,
                         logdir=None,
                         num_cores=None,
                         batch_size=128,
                         num_epochs=10,
                         num_workers=4,
                         log_steps=20,
                         lr=None,
                         momentum=None,
                         target_accuracy=None,
                         profiler_port=9012,
                         opts=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--datadir', type=str, default=datadir)
    parser.add_argument('--logdir', type=str, default=logdir)
    parser.add_argument('--num_cores', type=int, default=num_cores)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--num_epochs', type=int, default=num_epochs)
    parser.add_argument('--num_workers', type=int, default=num_workers)
    parser.add_argument('--log_steps', type=int, default=log_steps)
    parser.add_argument('--profiler_port', type=int, default=profiler_port)
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--momentum', type=float, default=momentum)
    parser.add_argument('--target_accuracy', type=float, default=target_accuracy)
    parser.add_argument('--drop_last', action='store_true')
    parser.add_argument('--fake_data', action='store_true')
    parser.add_argument('--tidy', action='store_true')
    parser.add_argument('--metrics_debug', action='store_true')
    parser.add_argument('--async_closures', action='store_true')
    parser.add_argument('--debug', action='store_true')
    if opts:
        for name, aopts in opts:
            parser.add_argument(name, **aopts)
    args, leftovers = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + leftovers
    # Setup import folders.
    xla_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
    sys.path.append(os.path.join(os.path.dirname(xla_folder), 'test'))
    return args

FLAGS = parse_common_options(
    datadir='/tmp/mnist-data',
    batch_size=128,
    momentum=0.5,
    lr=0.01,
    target_accuracy=98.0,
    num_epochs=18,
    opts=MODEL_OPTS.items())

class MNIST(nn.Module):

    def __init__(self):
        super(MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def _train_update(device, epoch, x, loss, tracker, writer):
  test_utils.print_training_update(
      device,
      x,
      loss.item(),
      tracker.rate(),
      tracker.global_rate(),
      epoch,
      summary_writer=writer)

def train_mnist(flags, fabric, **kwargs):
    torch.manual_seed(1)
    if flags.fake_data:
        train_loader = xu.SampleGenerator(
            data=(torch.zeros(flags.batch_size, 1, 28,
                            28), torch.zeros(flags.batch_size,
                                            dtype=torch.int64)),
        sample_count=60000 // flags.batch_size // xm.xrt_world_size())
        test_loader = xu.SampleGenerator(
            data=(torch.zeros(flags.batch_size, 1, 28,
                            28), torch.zeros(flags.batch_size,
                                            dtype=torch.int64)),
        sample_count=10000 // flags.batch_size // xm.xrt_world_size())
    else:
        train_dataset = datasets.MNIST(
            os.path.join(flags.datadir, str(xm.get_ordinal())),
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]))
        test_dataset = datasets.MNIST(
            os.path.join(flags.datadir, str(xm.get_ordinal())),
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]))
        train_sampler = None
        if xm.xrt_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=flags.batch_size,
            sampler=train_sampler,
            drop_last=flags.drop_last,
            shuffle=False if train_sampler else True,
            num_workers=flags.num_workers)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=flags.batch_size,
            drop_last=flags.drop_last,
            shuffle=False,
            num_workers=flags.num_workers)

    # Scale learning rate to num cores
    lr = flags.lr * fabric.world_size

    device = xm.xla_device()
    model = MNIST()

    model = fabric.setup(model)

    writer = None
    if xm.is_master_ordinal():
        writer = test_utils.get_summary_writer(flags.logdir)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=flags.momentum)

    optimizer = fabric.setup_optimizers(optimizer)
    loss_fn = nn.NLLLoss()

    def train_loop_fn(model, loader, epoch):
        tracker = xm.RateTracker()
        model.train()
        for step, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            fabric.backward(loss)
            optimizer.step()  # do not reduce gradients on sharded params
            tracker.add(flags.batch_size)
            if step % flags.log_steps == 0:
                xm.add_step_closure(
                    _train_update,
                    args=(device, epoch, step, loss, tracker, writer),
                    run_async=FLAGS.async_closures)

    def test_loop_fn(model, loader):
        total_samples = 0
        correct = 0
        model.eval()
        for data, target in loader:
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total_samples += data.size()[0]

        accuracy = 100.0 * correct.item() / total_samples
        accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
        return accuracy

    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    test_device_loader = pl.MpDeviceLoader(test_loader, device)
    accuracy, max_accuracy = 0.0, 0.0
    start_time = time.time()
    for epoch in range(1, flags.num_epochs + 1):
        fabric.print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
        train_loop_fn(model, train_device_loader, epoch)
        fabric.print('Epoch {} train end {}'.format(epoch, test_utils.now()))

        # TODO(alanwaketan): Investigate why inference would impact
        # the next epoch's training.
        with torch.no_grad():
            accuracy = test_loop_fn(model, test_device_loader)
        fabric.print('Epoch {} test end {}, Accuracy={:.2f}'.format(
            epoch, test_utils.now(), accuracy))
        max_accuracy = max(accuracy, max_accuracy)
        test_utils.write_to_summary(
            writer,
            epoch,
            dict_to_write={'Accuracy/test': accuracy},
            write_xla_metrics=True)
        if flags.metrics_debug:
            fabric.print(met.metrics_report())
    end_time = time.time()
    if flags.ckpt_consolidation:
        # Note: to run this test, all the model checkpoints needs to be
        # accessible from the master rank. Set --ckpt_prefix to a shared file
        # system (e.g. NFS) when running on a TPU pod.

        # Save the final model checkpoint
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()
        ckpt_path = f'{flags.ckpt_prefix}_rank-{rank:08d}-of-{world_size:08d}.pth'

        ckpt = {
            'model': model._forward_module.state_dict(),
            'shard_metadata': model._forward_module.get_shard_metadata(),
            'optimizer': optimizer.state_dict(),  # not needed in ckpt consolidation
        }
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        fabric.save(ckpt_path, ckpt)

        # Consolidate the sharded model checkpoints and test its accuracy
        if xm.is_master_ordinal(local=False):
            consolidate_sharded_model_checkpoints(
                ckpt_prefix=flags.ckpt_prefix, ckpt_suffix="_rank-*-of-*.pth")
        xm.rendezvous('ckpt_consolidation')
        model = MNIST().to(device)
        ckpt_consolidated = torch.load(f'{flags.ckpt_prefix}_consolidated.pth')
        model.load_state_dict(ckpt_consolidated['model'])
        accuracy = test_loop_fn(model, test_device_loader)
        xm.master_print(
            f'Checkpoint consolidated, Accuracy={accuracy:.2f} '
            '(note: it can be slightly different from the final training accuracy '
            'due to non-sync BatchNorm2d in the model)')

    test_utils.close_summary_writer(writer)
    fabric.print('Total training time: {:.2f}s'.format(end_time-start_time))
    fabric.print('Max Accuracy: {:.2f}%'.format(max_accuracy))
    return max_accuracy


def main(flags):

    # Automatic wrapping sub-modules with inner FSDP
    auto_wrap_policy = None
    auto_wrapper_callable = None
    if flags.auto_wrap_policy != "none":
        if flags.auto_wrap_policy == "size_based":
            # auto-wrap all sub-modules with a certain number of parameters (default 1000)
            # (in practice, one should set a larger min_num_params such as 1e8)
            auto_wrap_policy = partial(
                size_based_auto_wrap_policy,
                min_num_params=flags.auto_wrap_min_num_params)
        elif flags.auto_wrap_policy == "type_based":
            # auto-wrap all nn.Conv2d and nn.Linear sub-modules as an example
            # (transformer_auto_wrap_policy wraps all sub-modules in transformer_layer_cls)
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={nn.Conv2d, nn.Linear})
        else:
            raise Exception(f"Invalid auto-wrap policy: {flags.auto_wrap_policy}")
        if flags.use_gradient_checkpointing:
            # Apply gradient checkpointing to auto-wrapped sub-modules if specified
            auto_wrapper_callable = lambda m, *args, **kwargs: XLAFSDPStrategy(
                checkpoint_module(m), *args, **kwargs)

    strategy = XLAFSDPStrategy(
        compute_dtype=getattr(torch, flags.compute_dtype),
        fp32_reduce_scatter=flags.fp32_reduce_scatter,
        flatten_parameters=flags.flatten_parameters,
        shard_param_on_dim_0=flags.shard_param_on_dim_0,
        pin_layout_in_collective_ops=flags.pin_layout_in_collective_ops,
        auto_wrap_policy=auto_wrap_policy,
        auto_wrapper_callable=auto_wrapper_callable
    )

    fabric = L.Fabric(accelerator="tpu", devices=4, strategy=strategy)
    p_train_mnist = partial(train_mnist, flags)

    fabric.launch(p_train_mnist)


if __name__ == "__main__":
    main(FLAGS)