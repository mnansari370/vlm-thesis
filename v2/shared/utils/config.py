import copy
import os
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    Values from override take precedence over base.
    """
    merged = copy.deepcopy(base)

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)

    return merged


def _load_yaml_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a dictionary at top level: {path}")
    return data


def load_config(path: str) -> Dict[str, Any]:
    """
    Load YAML config.

    Supports optional _base_ inheritance:
      _base_:
        - ../base/file1.yaml
        - ../base/file2.yaml

    Child config overrides base config values.
    """
    path = os.path.abspath(path)
    cfg = _load_yaml_file(path)

    base_files = cfg.pop("_base_", None)
    if base_files is None:
        return cfg

    if isinstance(base_files, str):
        base_files = [base_files]

    if not isinstance(base_files, list):
        raise ValueError("_base_ must be a string or list of strings")

    merged_base: Dict[str, Any] = {}
    config_dir = os.path.dirname(path)

    for base_file in base_files:
        base_path = os.path.join(config_dir, base_file)
        base_cfg = load_config(base_path)
        merged_base = _deep_merge(merged_base, base_cfg)

    return _deep_merge(merged_base, cfg)