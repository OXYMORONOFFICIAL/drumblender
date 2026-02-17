"""
Audio datasets
"""
from collections import defaultdict
import json
import logging
import os
from pathlib import Path
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import random_split


# Setup logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class AudioDataset(Dataset):
    """
    Dataset of audio files.

    Args:
        data_dir: Path to the directory containing the dataset.
        meta_file: Name of the json metadata file.
        sample_rate: Expected sample rate of the audio files.
        num_samples: Expected number of samples in the audio files.
        split (optional): Split to return. Must be one of 'train', 'val', or 'test'.
            If None, the entire dataset is returned.
        seed: Seed for random number generator used to split the dataset.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        meta_file: str,
        sample_rate: int,
        # num_samples: int,
        num_samples: Optional[int], # allow variable length audio - xXx
        split: Optional[str] = None,
        seed: int = 42,
        split_strategy: Literal["sample_pack", "random"] = "random",
        normalize: bool = False,
        sample_types: Optional[List[str]] = None,
        instruments: Optional[List[str]] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.meta_file = meta_file
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.seed = seed
        self.normalize = normalize
        self.sample_types = sample_types
        self.instruments = instruments

        # Confirm that preprocessed dataset exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Preprocessed dataset not found. Expected: {self.data_dir}"
            )

        # Load the metadata
        with open(self.data_dir.joinpath(self.meta_file), "r") as f:
            self.metadata = json.load(f)

        self.file_list = sorted(self.metadata.keys())

        # Split the dataset
        if split is not None:
            if split_strategy == "sample_pack":
                self._sample_pack_split(split)
            elif split_strategy == "random":
                self._random_split(split)
            else:
                raise ValueError(
                    "Invalid split strategy. Expected one of 'sample_pack' or 'random'."
                )

    def __len__(self):
        return len(self.file_list)

    # def __getitem__(self, idx) -> Tuple[torch.Tensor]:
    #     audio_filename = self.metadata[self.file_list[idx]]["filename"]
    #     waveform, sample_rate = torchaudio.load(self.data_dir.joinpath(audio_filename))

    #     # Confirm sample rate and shape
    #     assert sample_rate == self.sample_rate, "Sample rate mismatch."
    #     assert waveform.shape == (1, self.num_samples), "Incorrect input audio shape."

    #     # Apply peak normalization
    #     if self.normalize:
    #         waveform = waveform / waveform.abs().max()

    #     return (waveform,)

    ###
    def __getitem__(self, idx) -> Tuple[torch.Tensor]:
        audio_filename = self.metadata[self.file_list[idx]]["filename"]
        waveform, sample_rate = torchaudio.load(self.data_dir.joinpath(audio_filename))

        # Ensure sample rate matches configuration.
        assert sample_rate == self.sample_rate, "Sample rate mismatch."

        # Expect [channels, time] and enforce mono by taking channel 0.
        if waveform.ndim != 2:
            raise ValueError(f"Expected (C,T), got {waveform.shape}")

        if waveform.shape[0] > 1:
            # Mono-only policy for all training/evaluation paths.
            waveform = waveform[:1, :]

        length = waveform.shape[-1]

        # Fixed-length mode: pad/truncate to num_samples.
        if self.num_samples is not None and self.num_samples > 0:
            if length > self.num_samples:
                waveform = waveform[:, : self.num_samples]
                length = self.num_samples
            elif length < self.num_samples:
                waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - length))
                length = self.num_samples

            # Preserve the original fixed-shape invariant.
            assert waveform.shape == (1, self.num_samples), "Incorrect input audio shape."
        else:
            # Variable-length mode only enforces mono-channel shape.
            assert waveform.shape[0] == 1, "Expecting mono audio"

        if self.normalize:
            waveform = waveform / (waveform.abs().max() + 1e-9)

        # Return waveform and length for collate/bucketing logic.
        return (waveform, length)
    ###


    def _sample_pack_split(
        self, split: str, test_size: float = 0.1, val_size: float = 0.1
    ):
        split_metadata = self._sample_pack_split_metadata(split, test_size, val_size)

        # Convert to list for file list
        self.file_list = split_metadata.index.tolist()
        log.info(f"Number of samples in {split} set: {len(self.file_list)}")

    def _sample_pack_split_metadata(
        self, split: str, test_size: float = 0.1, val_size: float = 0.1
    ):
        """
        Split the dataset into train, validation, and test sets. This creates splits
        that are disjont with respect to sample packs and have same the proportion of
        sample types. It performsn a greedy assignment of samples to splits, starting
        with the test set, then the validation set, and finally the training set.

        Args:
            split: Split to return. Must be one of 'train', 'val', or 'test'.
        """
        if split not in ["train", "val", "test"]:
            raise ValueError("Invalid split. Must be one of 'train', 'val', or 'test'.")

        data = pd.DataFrame.from_dict(self.metadata, orient="index")

        # Count the number of samples in each type (e.g. electric, acoustic)
        data_types = data.groupby("type").size().reset_index(name="counts")
        # log.info(f"Number of samples by type:\n {data_types}")

        # Filter by sample types
        if self.sample_types is not None:
            data_types = data_types[data_types["type"].isin(self.sample_types)]
            log.info(f"Filtering by sample types: {self.sample_types}")

        for t in data_types.iterrows():
            num_samples = t[1]["counts"]
            sample_type = t[1]["type"]

            # Group the samples by sample pack with counts and shuffle
            sample_packs = (
                data[data["type"] == sample_type]
                .groupby("sample_pack_key")
                .size()
                .reset_index(name="counts")
                .sample(frac=1, random_state=self.seed)
            )

            # Add a column for split
            sample_packs["split"] = "train"

            # Starting with the test set, greedily assign samples to splits -- if a
            # sample pack has fewer samples than the number of samples needed for the
            # split, assign all of the samples in the pack to the split.
            for s, n in zip(("test", "val"), (test_size, val_size)):
                split_samples = int(num_samples * n)
                for i, row in sample_packs.iterrows():
                    if row["counts"] <= split_samples and row["split"] == "train":
                        split_samples -= row["counts"]
                        sample_packs.loc[i, "split"] = s

            # Assign the split to the samples in data
            for i, row in sample_packs.iterrows():
                data.loc[
                    data["sample_pack_key"] == row["sample_pack_key"],
                    "split",
                ] = row["split"]

        # Count the number of samples in each split, log as percentage of total
        splits = data.groupby("split").size().reset_index(name="counts")
        splits["percent"] = splits["counts"] / splits["counts"].sum()
        log.info(f"Split counts:\n{splits}")

        # Filter by instrument types if specified
        if "instrument" in data.columns:
            log.info(f"Insrumens in dataset: {data['instrument'].unique()}")
            if self.instruments is not None:
                log.info(f"Filtering by instruments: {self.instruments}")
                data = data[data["instrument"].isin(self.instruments)]

        # Filter by split
        data = data[data["split"] == split]

        # Logging
        data_types = data.groupby("type").size().reset_index(name="counts")
        log.info(f"Number of samples by type:\n {data_types}")

        if "instrument" in data.columns:
            inst_types = data.groupby("instrument").size().reset_index(name="counts")
            log.info(f"Number of samples by instrument:\n {inst_types}")

        return data

    def _random_split(self, split: str):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            split: Split to return. Must be one of 'train', 'val', or 'test'.
        """
        if self.sample_types is not None:
            raise NotImplementedError(
                "Cannot use sample types with random split. Use sample_pack split."
            )

        if split not in ["train", "val", "test"]:
            raise ValueError("Invalid split. Must be one of 'train', 'val', or 'test'.")

        splits = random_split(
            self.file_list,
            [0.8, 0.1, 0.1],
            generator=torch.Generator().manual_seed(self.seed),
        )

        # Set the file list based on the split
        if split == "train":
            self.file_list = splits[0]
        elif split == "val":
            self.file_list = splits[1]
        elif split == "test":
            self.file_list = splits[2]


# class AudioWithParametersDataset(AudioDataset):
#     """
#     Dataset of audio pairs with an additional parameter tensor

#     Args:
#         data_dir: Path to the directory containing the dataset.
#         meta_file: Name of the json metadata file.
#         sample_rate: Expected sample rate of the audio files.
#         num_samples: Expected number of samples in the audio files.
#         parameter_ky: Key in the metadata file for the feature file.
#         **kwargs: Additional arguments to pass to AudioPairDataset.
#     """

#     def __init__(
#         self,
#         data_dir: Union[str, Path],
#         meta_file: str,
#         sample_rate: int,
#         num_samples: int,
#         parameter_key: str,
#         expected_num_modes: Optional[int] = None,
#         **kwargs,
#     ):
#         super().__init__(
#             data_dir=data_dir,
#             meta_file=meta_file,
#             sample_rate=sample_rate,
#             num_samples=num_samples,
#             **kwargs,
#         )
#         self.parameter_key = parameter_key
#         self.expected_num_modes = expected_num_modes

#     def __getitem__(self, idx):
#         (waveform_a,) = super().__getitem__(idx)
#         feature_file = self.metadata[self.file_list[idx]][self.parameter_key]
#         feature = torch.load(self.data_dir.joinpath(feature_file))

#         # Pad with zeros if the number of modes is less than expected
#         if (
#             self.expected_num_modes is not None
#             and feature.shape[1] != self.expected_num_modes
#         ):
#             null_features = torch.zeros(
#                 (
#                     feature.shape[0],
#                     self.expected_num_modes - feature.shape[1],
#                     feature.shape[2],
#                 )
#             )
#             feature = torch.cat((feature, null_features), dim=1)

#         return waveform_a, feature



###
class AudioWithParametersDataset(Dataset):
    """
    Loads (waveform, params, length).

    waveform: [1, T]
    params:   [P, M, F] where P is typically 3 (freq, amp, phase)
    length:   int (T)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        meta_file: str,
        sample_rate: int,
        num_samples: Optional[int],
        split: Optional[str] = None,
        seed: int = 42,
        split_strategy: Literal["sample_pack", "random"] = "random",
        normalize: bool = False,
        sample_types: Optional[List[str]] = None,
        instruments: Optional[List[str]] = None,
        parameter_key: str = "feature_file",
        expected_num_modes: Optional[int] = None,
        split_train_ratio: float = 0.8,
        split_val_ratio: float = 0.1,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.meta_file = meta_file
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.seed = seed
        self.split = split
        self.split_strategy = split_strategy
        self.normalize = normalize
        self.sample_types = sample_types
        self.instruments = instruments
        self.parameter_key = parameter_key
        self.expected_num_modes = expected_num_modes
        self.split_train_ratio = split_train_ratio
        self.split_val_ratio = split_val_ratio

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset dir not found: {self.data_dir}")

        with open(self.data_dir.joinpath(self.meta_file), "r") as f:
            self.metadata = json.load(f)

        # Keep deterministic ordering before filtering/splitting.
        self.file_list = sorted(self.metadata.keys())

        # Apply optional metadata filters first.
        if sample_types is not None:
            self.file_list = [
                k
                for k in self.file_list
                if ("type" in self.metadata[k] and self.metadata[k]["type"] in sample_types)
            ]
        if instruments is not None:
            self.file_list = [
                k
                for k in self.file_list
                if (
                    "instrument" in self.metadata[k]
                    and self.metadata[k]["instrument"] in instruments
                )
            ]

        # HIGHLIGHT: Train/val/test split is now created at dataset-load time
        # from sample_pack_key and seed. Preprocessing only needs pack metadata.
        if split is not None:
            if split_strategy == "sample_pack":
                self.file_list = self._split_within_pack(
                    keys=self.file_list,
                    split=split,
                    train_ratio=self.split_train_ratio,
                    val_ratio=self.split_val_ratio,
                )
            elif split_strategy == "random":
                self.file_list = self._split_random(
                    keys=self.file_list,
                    split=split,
                    train_ratio=self.split_train_ratio,
                    val_ratio=self.split_val_ratio,
                )
            else:
                raise ValueError(
                    "Invalid split strategy. Expected one of 'sample_pack' or 'random'."
                )

        # Cache lengths for bucketing.
        self.lengths = []
        for k in self.file_list:
            item = self.metadata[k]
            if "num_samples" in item:
                self.lengths.append(int(item["num_samples"]))
            else:
                wav_path = self.data_dir.joinpath(item["filename"])
                try:
                    info = torchaudio.info(wav_path)
                    self.lengths.append(int(info.num_frames))
                except Exception:
                    w, _ = torchaudio.load(wav_path)
                    self.lengths.append(int(w.shape[-1]))

    def __len__(self):
        return len(self.file_list)

    def _split_within_pack(
        self,
        keys: List[str],
        split: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> List[str]:
        if split not in ("train", "val", "test"):
            raise ValueError("Invalid split. Must be one of 'train', 'val', or 'test'.")
        if train_ratio < 0.0 or val_ratio < 0.0 or (train_ratio + val_ratio) > 1.0:
            raise ValueError(
                "Invalid split ratios: require train >= 0, val >= 0, train+val <= 1."
            )

        by_pack = defaultdict(list)
        for key in keys:
            pack_key = self.metadata[key].get("sample_pack_key", "__root__")
            by_pack[pack_key].append(key)

        g = torch.Generator().manual_seed(self.seed)
        selected: List[str] = []

        for pack_key in sorted(by_pack.keys()):
            pack_items = sorted(by_pack[pack_key])
            perm = torch.randperm(len(pack_items), generator=g).tolist()
            shuffled = [pack_items[i] for i in perm]

            n = len(shuffled)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            # Ensure each pack contributes at least one train sample when possible.
            if n > 0 and n_train == 0:
                n_train = 1
            if n_train + n_val > n:
                n_val = max(0, n - n_train)

            if split == "train":
                selected.extend(shuffled[:n_train])
            elif split == "val":
                selected.extend(shuffled[n_train : n_train + n_val])
            else:
                selected.extend(shuffled[n_train + n_val :])

        log.info(
            "AudioWithParametersDataset split='%s' strategy='sample_pack' size=%d",
            split,
            len(selected),
        )
        return selected

    def _split_random(
        self,
        keys: List[str],
        split: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> List[str]:
        if split not in ("train", "val", "test"):
            raise ValueError("Invalid split. Must be one of 'train', 'val', or 'test'.")
        if train_ratio < 0.0 or val_ratio < 0.0 or (train_ratio + val_ratio) > 1.0:
            raise ValueError(
                "Invalid split ratios: require train >= 0, val >= 0, train+val <= 1."
            )

        keys = sorted(keys)
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(len(keys), generator=g).tolist()
        shuffled = [keys[i] for i in perm]

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if n > 0 and n_train == 0:
            n_train = 1
        if n_train + n_val > n:
            n_val = max(0, n - n_train)

        if split == "train":
            selected = shuffled[:n_train]
        elif split == "val":
            selected = shuffled[n_train : n_train + n_val]
        else:
            selected = shuffled[n_train + n_val :]

        log.info(
            "AudioWithParametersDataset split='%s' strategy='random' size=%d",
            split,
            len(selected),
        )
        return selected

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int]:
        key = self.file_list[idx]
        item = self.metadata[key]

        wav_path = self.data_dir.joinpath(item["filename"])
        waveform, sr = torchaudio.load(wav_path)
        assert sr == self.sample_rate, f"Sample rate mismatch: {sr} != {self.sample_rate}"

        # Mono-only policy: always use channel 0.
        waveform = waveform[:1, :]
        length = int(waveform.shape[-1])

        # Fixed-length mode: pad or truncate.
        if self.num_samples is not None and self.num_samples > 0:
            target = int(self.num_samples)
            if length > target:
                waveform = waveform[:, :target]
                length = target
            elif length < target:
                waveform = torch.nn.functional.pad(waveform, (0, target - length))
                length = target

        if self.normalize:
            waveform = waveform / (waveform.abs().max() + 1e-9)

        if self.parameter_key not in item:
            raise KeyError(f"metadata missing '{self.parameter_key}' for key={key}")

        p_path = self.data_dir.joinpath(item[self.parameter_key])
        params = torch.load(p_path)
        if params.ndim != 3:
            raise ValueError(
                f"Expected params [P,M,F], got {tuple(params.shape)} from {p_path}"
            )

        if self.expected_num_modes is not None:
            m_expected = int(self.expected_num_modes)
            p_dim, m_dim, f_dim = params.shape
            if m_dim > m_expected:
                params = params[:, :m_expected, :]
            elif m_dim < m_expected:
                pad = params.new_zeros((p_dim, m_expected - m_dim, f_dim))
                params = torch.cat([params, pad], dim=1)

        return waveform, params, length
###
