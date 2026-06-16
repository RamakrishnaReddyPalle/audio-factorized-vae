# src/dataset/bucket_sampler.py

import random

from torch.utils.data import Sampler


class BucketBatchSampler(Sampler):

    def __init__(
        self,
        dataset,
        batch_size,
        drop_last=False
    ):

        self.dataset = dataset

        self.batch_size = (
            batch_size
        )

        self.drop_last = (
            drop_last
        )

        self.buckets = [

            (0, 40),

            (40, 60),

            (60, 80),

            (80, 100),

            (100, 1000)
        ]

        self.bucket_indices = []

        lengths = []

        for idx in range(
            len(dataset)
        ):

            lengths.append(

                dataset[idx][
                    "length"
                ]
            )

        for low, high in self.buckets:

            bucket = [

                idx

                for idx, length

                in enumerate(
                    lengths
                )

                if (
                    low
                    <=
                    length
                    <
                    high
                )
            ]

            self.bucket_indices.append(
                bucket
            )

    def __iter__(self):

        all_batches = []

        for bucket in self.bucket_indices:

            random.shuffle(
                bucket
            )

            for i in range(

                0,

                len(bucket),

                self.batch_size
            ):

                batch = bucket[
                    i:
                    i+self.batch_size
                ]

                if (

                    len(batch)
                    <
                    self.batch_size

                    and

                    self.drop_last
                ):

                    continue

                all_batches.append(
                    batch
                )

        random.shuffle(
            all_batches
        )

        return iter(
            all_batches
        )

    def __len__(self):

        total = 0

        for bucket in self.bucket_indices:

            total += (

                len(bucket)
                //
                self.batch_size
            )

            if (

                not self.drop_last

                and

                len(bucket)
                %
                self.batch_size
                != 0
            ):

                total += 1

        return total