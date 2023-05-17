import argparse
import os

import deepspeed
import torch
import torch.distributed



def main():
    config = {
        "train_micro_batch_size_per_gpu": 2,
        "zero_optimization": {"stage": 3},
    }

    os.environ["LOCAL_RANK"] = "0"
    torch.distributed.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:12345", world_size=1, rank=0)

    with deepspeed.zero.Init(config=config):
        model = torch.nn.Linear(32, 2)

    engine, _, _, _ = deepspeed.initialize(args=argparse.Namespace(device_rank=0), model=model, config=config)
    input = torch.rand(2, 32).to("cuda:0")
    loss = engine(input).sum()

    optimizer = torch.optim.Adam(model.parameters())
    engine, opt, _, _ = deepspeed.initialize(
        args=argparse.Namespace(device_rank=0), model=model, optimizer=optimizer, config=config
    )
    input = torch.rand(2, 32).to("cuda:0")
    loss = engine(input).sum()
    engine.backward(loss)

    # trainer = Trainer(
    #     strategy=DeepSpeedStrategy(stage=3),
    #     accelerator="cuda",
    #     devices=2,
    #     fast_dev_run=True,
    #     precision="16-mixed",
    # )
    # trainer.test(model)
    # print(model.ds_inflight_param_registry)
    # trainer.fit(model)


if __name__ == "__main__":
    main()
