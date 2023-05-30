"""Utilities that add functionality to yahp Hparams."""
from __future__ import annotations

import dataclasses
from enum import Enum
from typing import List

import yahp as hp


def stringify_field(field: bool | int | float | str | Enum | List | hp.Hparams) -> str:
    """Generate the string representation of a Hparams field.

    If the field is an Enum, List, or Hparams, recursively apply stringify_field to its value(s).
    If it is a bool, int, or float, cast it to a str. If it is a str, return as is.

    Args:
        field (bool | int | float | str | Enum | List | hp.Hparams): Hparams field to stringify.

    Raises:
        TypeError: If field does not have the correct type for a Hparams field.

    Returns:
        str: String representation of field.
    """
    if isinstance(field, hp.Hparams):
        # If field implements description, use field.description, else use the default description for hparams
        field_name = getattr(field, "description", description(field))
    elif isinstance(field, List):
        field_name = f"[{', '.join([stringify_field(item) for item in field])}]"
    elif isinstance(field, Enum):
        field_name = stringify_field(field.value)
    elif isinstance(field, float) or isinstance(field, int) or isinstance(field, bool):
        field_name = str(field)
    elif isinstance(field, str):
        field_name = field
    else:
        raise TypeError(f"Field {field} must be of type bool, int, float, str, Enum, List, or Hparams")
    return field_name


def description(hparams: hp.Hparams) -> str:
    """Generate the string representation for hparams.

    The Hparams description is generated from the dataclass fields of hparams. First, all non-identifying fields (if
    they exist) that should not be a part of the description are removed. Then, fields with a value of None are removed.
    Finally, a string representation of each field is generated and they are combined with the class name of hparams.

    Args:
        hparams (hp.Hparams): Object to generate the description of.

    Returns:
        str: Description of hparams.
    """
    field_names = [field.name for field in dataclasses.fields(hparams)]
    non_id_fields = getattr(hparams, "non_id_fields", [])
    field_names = [field_name for field_name in field_names if field_name not in non_id_fields]
    field_names = [field_name for field_name in field_names if getattr(hparams, field_name) is not None]
    field_names = [f"{field_name}={stringify_field(getattr(hparams, field_name))}" for field_name in field_names]
    hparam_name = f"{type(hparams).__name__}({', '.join(field_names)})"
    return hparam_name
