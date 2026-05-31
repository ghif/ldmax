"""Configuration utilities using ml_collections."""

import yaml
from ml_collections import ConfigDict

def load_config(path: str) -> ConfigDict:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        A ConfigDict instance.
    """
    with open(path, "r") as f:
        config_data = yaml.safe_load(f)
    return ConfigDict(config_data)

def save_config(config: ConfigDict, path: str):
    """Save configuration to a YAML file.

    Args:
        config: The ConfigDict to save.
        path: Path to the output YAML file.
    """
    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f)
