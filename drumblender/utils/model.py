"""
Helpful utils for handling pre-trained models
"""
import inspect
from typing import Any

import yaml
from jsonargparse import ArgumentParser

from drumblender.data import AudioDataModule
from drumblender.tasks import DrumBlender


def load_model(config: str, ckpt: str, include_data: bool = False):
    """
    Load model from checkpoint
    """
    # Load the config file and instantiate the model
    config_parser = ArgumentParser()
    config_parser.add_subclass_arguments(DrumBlender, "model", fail_untyped=False)
    config_parser.add_argument("--trainer", type=dict, default={})
    config_parser.add_argument("--seed_everything", type=int)
    config_parser.add_argument("--ckpt_path", type=str)
    config_parser.add_argument("--optimizer", type=dict)
    config_parser.add_argument("--lr_scheduler", type=dict)

    if include_data:
        config_parser.add_subclass_arguments(AudioDataModule, "data")
        config = config_parser.parse_path(config)
    else:
        # ### HIGHLIGHT: Accept both dict-style and string-style top-level `data` entries.
        # Some configs use `data: data/percussion.yaml`, which should not block model-only loads.
        config_parser.add_argument("--data", type=Any, default={})
        with open(config, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
        if isinstance(raw_cfg, dict) and not isinstance(raw_cfg.get("data", {}), dict):
            raw_cfg["data"] = {}
        config = config_parser.parse_object(raw_cfg)
    init = config_parser.instantiate_classes(config)

    # Get the constructor arguments for the DrumBlender task and create a dictionary of
    # keyword arguments to instantiate a new DrumBlender object from checkpoint
    init_args = inspect.getfullargspec(DrumBlender.__init__).args
    model_dict = {
        attr: getattr(init.model, attr)
        for attr in init_args
        if attr != "self" and hasattr(init.model, attr)
    }

    # Load the checkpoint
    print(f"Loading checkpoint from {ckpt}...")

    # Load new model from checkpoint file
    model = init.model.load_from_checkpoint(ckpt, **model_dict)

    # Instantiate the datamodule if required
    if include_data:
        datamodule = init.data
        return model, datamodule

    return model, None


def load_datamodule(config: str):
    """
    Load a datamodule from a config file
    """
    datamodule_parser = ArgumentParser()
    datamodule_parser.add_subclass_arguments(AudioDataModule, "datamodule")
    if config is not None:
        with open(config, "r") as f:
            config = yaml.safe_load(f)
            config = {"datamodule": config}
            datamodule_args = datamodule_parser.parse_object(config)
            datamodule = datamodule_parser.instantiate_classes(
                datamodule_args
            ).datamodule

    return datamodule


def load_config_yaml(config: str):
    """
    Load a config file
    """
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    return config
