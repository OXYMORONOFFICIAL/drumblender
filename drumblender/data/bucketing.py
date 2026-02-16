from __future__ import annotations
from typing import Iterable, List, Sequence

import torch
from torch.utils.data import Sampler


class BucketingBatchSampler(Sampler[List[int]]):
    """
    길이 기반 버킷팅 배치 샘플러.
    - lengths: 각 샘플의 길이(샘플 수)
    - boundaries: 버킷 경계(샘플 수). 예: [48000, 96000, 192000, 384000]
      => (<=48000), (48000~96000], (96000~192000], (192000~384000], (>384000)
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        boundaries: Sequence[int],
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.lengths = list(map(int, lengths))
        self.batch_size = int(batch_size)
        self.boundaries = list(map(int, boundaries))
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # indices -> bucket_id
        self.buckets = [[] for _ in range(len(self.boundaries) + 1)]
        for i, L in enumerate(self.lengths):
            b = 0
            while b < len(self.boundaries) and L > self.boundaries[b]:
                b += 1
            self.buckets[b].append(i)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self) -> Iterable[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # bucket 내부 shuffle
        bucket_lists = []
        for bucket in self.buckets:
            if len(bucket) == 0:
                continue
            bucket = bucket.copy()
            if self.shuffle:
                perm = torch.randperm(len(bucket), generator=g).tolist()
                bucket = [bucket[i] for i in perm]
            bucket_lists.append(bucket)

        # bucket별로 batch 만들기
        batches: List[List[int]] = []
        for bucket in bucket_lists:
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        # batch 순서 shuffle
        if self.shuffle:
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]

        yield from batches

    def __len__(self) -> int:
        total = 0
        for bucket in self.buckets:
            if self.drop_last:
                total += len(bucket) // self.batch_size
            else:
                total += (len(bucket) + self.batch_size - 1) // self.batch_size
        return total
