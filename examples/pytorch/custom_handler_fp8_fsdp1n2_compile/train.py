import argparse
import logging
from dataclasses import dataclass

import lightning as L
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.demos import WikiText2
from lightning.pytorch.strategies import FSDPStrategy, ModelParallelStrategy
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


@dataclass
class Args:
    vocab_size: int = 32000
    enable_fp8: bool = False
    enable_torch_compile: bool = False
    enable_cpu_offload: bool = False
    enable_gradient_checkpointing: bool = False
    enable_fsdp2: bool = False


class SimpleLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        print(f"Input shape before Linear: {x.shape}")
        x = self.linear(x)
        print(f"Output shape after Linear: {x.shape}")
        return self.activation(x)


class InnerModel(nn.Module):
    def __init__(self, num_layers, hidden_size, vocab_size=32000):
        super().__init__()
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        # Initialize a ModuleList to store the intermediate layers
        self.layers = nn.ModuleList([SimpleLayer(hidden_size) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        # Pass the input through each layer sequentially
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # The wrapped Transformer model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class LanguageModel(L.LightningModule):
    def __init__(
        self,
        vocab_size=32000,
        enable_fp8=False,
        enable_fsdp2=False,
        enable_torch_compile=False,
        enable_gradient_checkpointing=False,
        enable_cpu_offload=False,
    ):
        super().__init__()
        self.model = None
        self.vocab_size = vocab_size
        self.enable_fp8 = enable_fp8
        self.enable_fsdp2 = enable_fsdp2
        self.enable_torch_compile = enable_torch_compile
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_cpu_offload = enable_cpu_offload
        self.model_path = "dummy"  # placeholder
        self.parallel_dims = {"dp_shard_enabled": torch.cuda.device_count() > 1}  # only used for FP8 training

    def log_model_stage(self, stage: str):
        """Logs the current state of the model with a description of the stage.

        Args:
            stage (str): Description of the current model stage.

        """
        log.warning(f"Model at stage: {stage}\n{self.model}")

    def configure_torch_compile(self):
        if self.enable_torch_compile:
            from handlers.torch_compile_handler import TorchCompileHandler

            torch_compile_handler = TorchCompileHandler(
                enable_compile=self.enable_torch_compile,
                model_path=self.model_path,
                # Implicitly specify layers, default only support compile HuggingFace llama and mixtral model with llama MLP block and Mixtral MixtralBlockSparseTop2MLP block compiled
                compile_layers=["SimpleLayer"],
                compile_args=None,
            )
            torch_compile_handler.compile_model(self.model)

        self.log_model_stage("Model after compile")

    def configure_fsdp2(self):
        if self.enable_fsdp2:
            self.all_gpus = dist.new_group(backend="nccl")
            dp_mesh = self.device_mesh["data_parallel"]
            assert dp_mesh.size() > 1

            from handlers.fsdp2_handler import FSDP2Config, FSDP2Handler

            fsdp2_config = FSDP2Config(
                enable_cpu_offload=self.enable_cpu_offload,
                enable_gradient_checkpointing=self.enable_gradient_checkpointing,
            )
            fsdp2_handler = FSDP2Handler(fsdp2_config, self.device_mesh)
            self.model = fsdp2_handler.wrap_model(self.model)

        self.log_model_stage("Model after FSDP wrapper")

    def configure_fp8(self):
        # Setup fp8 training, if enable_fp8 is false, it will create a fake handler
        from handlers.fp8_training_handler import Float8TrainingHandler, FP8Config

        fp8_config = FP8Config(
            enable_fp8=self.enable_fp8,
            enable_amax_init=False,
            scaling_type_input="delayed",
            scaling_type_weight="delayed",
            scaling_type_grad_output="delayed",
            enable_fsdp_float8_all_gather=False,
            precompute_float8_dynamic_scale_for_fsdp=False,
            pad_inner_dim=True,
            emulate_fp8=False,  # Set to True for testing without FP8 hardware
            enable_torch_compile=self.enable_torch_compile,
            enable_pre_and_post_forward=False,
        )
        self.fp8_handler = Float8TrainingHandler(fp8_config, self.model_path, self.parallel_dims)
        self.fp8_handler.convert_to_float8_training(self.model)
        self.log_model_stage("Model after FP8 wrapper")

    def configure_model(self):
        if self.model is not None:
            return

        with torch.device("meta"):
            self.model = ModelWrapper(
                InnerModel(
                    num_layers=16,
                    hidden_size=1024,
                    vocab_size=self.vocab_size,
                )
            )
        self.configure_fp8()
        self.configure_fsdp2()
        self.configure_torch_compile()
        self.model.train()

    def on_train_batch_start(self, batch, batch_idx):
        super().on_train_batch_start(batch, batch_idx)
        self.hand_roll_base_zero_grad()

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        super().on_validation_batch_start(batch, batch_idx, dataloader_idx)
        self.hand_roll_base_zero_grad()

    def hand_roll_base_zero_grad(self):
        # to resolve the torch compile + FSDP1 issue https://github.com/pytorch/pytorch/issues/139110
        if self.enable_torch_compile and not self.enable_fsdp2:
            self.zero_grad(set_to_none=True)
            for p in self.parameters():
                if p._base is not None and p._base.grad is not None:
                    p._base._grad = None

    def on_before_optimizer_step(self, optimizer):
        self.fp8_handler.sync_float8_amax_and_scale_history(self.model)
        super().on_before_optimizer_step(optimizer)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.fp8_handler.precompute_float8_dynamic_scale_for_fsdp(self.model)
        super().on_train_batch_end(outputs, batch, batch_idx)

    def training_step(self, batch):
        input, target = batch
        output = self.model(input)
        log_softmax = nn.LogSoftmax(dim=1)
        loss = F.nll_loss(log_softmax(output).view(-1, self.vocab_size), target.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


def train(args):
    L.seed_everything(42)

    dataset = WikiText2()
    train_dataloader = DataLoader(dataset, num_workers=8, batch_size=1)

    model = LanguageModel(
        vocab_size=args.vocab_size,
        enable_fp8=args.enable_fp8,
        enable_fsdp2=args.enable_fsdp2,
        enable_torch_compile=args.enable_torch_compile,
        enable_gradient_checkpointing=args.enable_gradient_checkpointing,
        enable_cpu_offload=args.enable_cpu_offload,
    )

    if args.enable_fsdp2:
        strategy = ModelParallelStrategy(
            data_parallel_size=1,
            tensor_parallel_size=1,
        )
    else:
        layers = {SimpleLayer}
        strategy = FSDPStrategy(
            auto_wrap_policy=layers,
            sharding_strategy="FULL_SHARD",
            forward_prefetch=True,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            sync_module_states=True,
            activation_checkpointing_policy=layers if args.enable_gradient_checkpointing else None,
            # for FSDP, we set mixed precision here instead of passing precision to PL trainer.
            # precision="bf16-true" in PL trainer means pure half precision (including optimizer update etc.)
            # while precision="bf16-mixed" results in unshard allgather performed in fp32:
            # https://github.com/Lightning-AI/pytorch-lightning/blob/bf25167bbf64f50ba335aa759318946b21775cd2/src/lightning/fabric/plugins/precision/fsdp.py#L83
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
            cpu_offload=args.enable_cpu_offload,
        )
    trainer = L.Trainer(strategy=strategy, max_steps=100, precision="bf16-true", accumulate_grad_batches=8)

    trainer.fit(model, train_dataloader)

    trainer.print(torch.cuda.memory_summary())


def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model.")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size. Default is 32000.")
    parser.add_argument("--enable_fp8", action="store_true", help="Enable FP8 precision.")
    parser.add_argument("--enable_torch_compile", action="store_true", help="Enable Torch Compile.")
    parser.add_argument("--enable_cpu_offload", action="store_true", help="Enable CPU offload.")
    parser.add_argument("--enable_gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--enable_fsdp2", action="store_true", help="Enable FSDP2.")
    args = parser.parse_args()
    return Args(**vars(args))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    train(args)
