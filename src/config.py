"""
Configuration Loader for EGX Signals
Loads params.yaml and enhancement.yaml
"""
from pathlib import Path
from typing import Any, Dict
import yaml

_CONFIG_CACHE: Dict[str, Any] = {}

def load_params(config_path: str = None) -> Dict[str, Any]:
    """Load params.yaml with caching."""
    global _CONFIG_CACHE
    if "params" in _CONFIG_CACHE:
        return _CONFIG_CACHE["params"]

    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "params.yaml"

    with open(config_path, "r") as f:
        params = yaml.safe_load(f)

    _CONFIG_CACHE["params"] = params
    return params

def load_enhancement_config(config_path: str = None) -> Dict[str, Any]:
    """Load enhancement.yaml with caching."""
    global _CONFIG_CACHE
    if "enhancement" in _CONFIG_CACHE:
        return _CONFIG_CACHE["enhancement"]

    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "enhancement.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    _CONFIG_CACHE["enhancement"] = config
    return config

def reload_configs():
    """Force reload of all configs."""
    global _CONFIG_CACHE
    _CONFIG_CACHE.clear()
    load_params()
    load_enhancement_config()