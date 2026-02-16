"""
Audio datasets
"""
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

        self.file_list = list(self.metadata.keys())

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

        # SR? ?ъ쟾??媛뺤젣(沅뚯옣: ?꾩쿂由щ줈 ?듭씪)
        assert sample_rate == self.sample_rate, "Sample rate mismatch."

        # mono 媛뺤젣(?곗씠?곌? stereo?????덉쑝硫??ш린???ㅼ슫誘뱀뒪 沅뚯옣)
        # 湲곗〈 肄붾뱶泥섎읆 (1, T)留??덉슜?섎젮硫??꾨옒 assert ?좎?
        if waveform.ndim != 2:
            raise ValueError(f"Expected (C,T), got {waveform.shape}")

        if waveform.shape[0] > 1:
            # ?ㅼ슫誘뱀뒪(mean)
            waveform = waveform[:1, :]

        length = waveform.shape[-1]

        # 怨좎젙 湲몄씠瑜??먰븷 ?뚮쭔 寃???먮Ⅴ湲??⑤뵫
        if self.num_samples is not None and self.num_samples > 0:
            # pad/truncate ?댁꽌 怨좎젙 湲몄씠濡?留욎땄
            if length > self.num_samples:
                waveform = waveform[:, : self.num_samples]
                length = self.num_samples
            elif length < self.num_samples:
                waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - length))
                length = self.num_samples

            # 湲곗〈怨??숈씪??assert (?명솚)
            assert waveform.shape == (1, self.num_samples), "Incorrect input audio shape."
        else:
            # variable-length 紐⑤뱶?먯꽌??(1, T_i)留?蹂댁옣?섎㈃ ??
            assert waveform.shape[0] == 1, "Expecting mono audio"

        if self.normalize:
            waveform = waveform / (waveform.abs().max() + 1e-9)

        # length 媛숈씠 諛섑솚 (collate_fn???닿구 ?댁슜)
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
    params:   [P, M, F]  (?? P=3: freq/amp/phase, M=num_modes, F=num_frames)
    length:   int (T)

    Args:
        data_dir: preprocessed dataset root
        meta_file: json metadata (key -> dict with filename, feature_file, etc.)
        sample_rate: expected SR (assert)
        num_samples: if not None => pad/truncate to fixed length, else variable
        parameter_key: metadata?먯꽌 params 寃쎈줈濡???key (default "feature_file")
        expected_num_modes: 紐⑤뱶 ??怨좎젙?섍퀬 ?띠쑝硫?吏??(遺議깊븯硫?pad, 留롮쑝硫?truncate)
        normalize: peak normalize ?щ?
        split/split_strategy/seed/sample_types/instruments: AudioDataset怨??좎궗?섍쾶 ?꾪꽣留??듭뀡
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

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset dir not found: {self.data_dir}")

        with open(self.data_dir.joinpath(self.meta_file), "r") as f:
            self.metadata = json.load(f)

        # file_list = metadata keys
        self.file_list = list(self.metadata.keys())

        # ---- optional split/filter (媛꾨떒 援ы쁽: 湲곗〈 AudioDataset 濡쒖쭅??洹몃?濡?媛?몄삤怨??띠쑝硫?
        #      ?덇? ?대? AudioDataset 履?split?⑥닔?ㅼ쓣 媛뽮퀬 ?덉쑝??洹멸구 ?몄텧?대룄 ??
        # ?ш린?쒕뒗 "split"??metadata???대? ?덈떎硫?洹멸구 ?ъ슜?섎룄濡?媛???꾩떎??
        if split is not None:
            # metadata??"split" ?꾨뱶媛 ?덉쑝硫?洹멸구濡??꾪꽣
            if all("split" in self.metadata[k] for k in self.file_list):
                self.file_list = [k for k in self.file_list if self.metadata[k]["split"] == split]

        # filter by type/instrument if present
        if sample_types is not None:
            self.file_list = [
                k for k in self.file_list
                if ("type" in self.metadata[k] and self.metadata[k]["type"] in sample_types)
            ]
        if instruments is not None:
            self.file_list = [
                k for k in self.file_list
                if ("instrument" in self.metadata[k] and self.metadata[k]["instrument"] in instruments)
            ]

        # ---- lengths cache for bucketing ----
        # metadata??num_samples ??ν빐?먮㈃ 鍮좊Ⅴ怨? ?놁쑝硫?torchaudio.info濡??쒕쾲 ?ㅼ틪
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
                    # fallback: load (?먮┝)
                    w, _ = torchaudio.load(wav_path)
                    self.lengths.append(int(w.shape[-1]))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int]:
        key = self.file_list[idx]
        item = self.metadata[key]

        wav_path = self.data_dir.joinpath(item["filename"])
        waveform, sr = torchaudio.load(wav_path)  # [C,T]
        assert sr == self.sample_rate, f"Sample rate mismatch: {sr} != {self.sample_rate}"

        # downmix(mean)
        if waveform.shape[0] > 1:
            waveform = waveform[:1, :]
        else:
            waveform = waveform[:1, :]

        length = int(waveform.shape[-1])

        # fixed length 紐⑤뱶硫?pad/truncate
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

        # params 濡쒕뱶
        if self.parameter_key not in item:
            raise KeyError(f"metadata missing '{self.parameter_key}' for key={key}")

        p_path = self.data_dir.joinpath(item[self.parameter_key])
        params = torch.load(p_path)  # 湲곕?: [P,M,F]
        if params.ndim != 3:
            raise ValueError(f"Expected params [P,M,F], got {tuple(params.shape)} from {p_path}")

        # num_modes ?뺣━(?먰븯硫?
        if self.expected_num_modes is not None:
            M = int(self.expected_num_modes)
            P, M0, F = params.shape
            if M0 > M:
                params = params[:, :M, :]
            elif M0 < M:
                pad = params.new_zeros((P, M - M0, F))
                params = torch.cat([params, pad], dim=1)

        return waveform, params, length
###
