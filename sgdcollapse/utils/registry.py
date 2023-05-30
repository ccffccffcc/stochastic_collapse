"""Minimal implementation of registry: Dict with functionality abstracted away into utility functions."""

from typing import Dict, List

Registry = Dict


def options(registry: Registry) -> List[str]:
    """Get the keys for the classes registered in the registry.

    Args:
        registry (Registry): Queried registry.

    Returns:
        List[str]: List of keys for the classes registered in registry.
    """
    return list(registry.keys())
