#!/usr/bin/env python3
"""
Standalone NCCL + CUDA all_reduce smoke test.
"""

import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    x = torch.tensor([rank + 1.0], device="cuda")
    t0 = time.time()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    t1 = time.time()

    expected = world * (world + 1) / 2
    print(
        f"[rank {rank}] local_rank={local_rank} all_reduce={float(x.item())} "
        f"expected={expected} took={t1 - t0:.4f}s",
        flush=True,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

