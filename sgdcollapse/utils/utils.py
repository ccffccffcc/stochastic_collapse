import hashlib
import io
import os
import shutil
from typing import Callable, Optional, TypeVar

from composer.utils import ObjectStore, dist
from libcloud.storage.types import ObjectDoesNotExistError

import warnings
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from composer.loss.utils import ensure_targets_one_hot, infer_target_type

T = TypeVar("T")


def hash(string: str) -> str:
    """Generate a hash for string."""
    return hashlib.md5(string.encode("utf-8")).hexdigest()


def make_new_local_dir(dir_path: str) -> None:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def object_exists(object_name: str, object_store: Optional[ObjectStore]) -> bool:
    if object_store is None:
        return False
    try:
        object_store.get_object_size(object_name)
    except ObjectDoesNotExistError:
        return False
    return True


def artifact_exists(artifact_name: str, object_store: Optional[ObjectStore]) -> bool:
    if object_exists(artifact_name, object_store) or os.path.exists(artifact_name):
        return True
    return False


def save_object(
    artifact: T, location: str, name: str, object_store: ObjectStore, save_fn: Callable[[T, str], None]
) -> None:
    assert object_store is not None
    save_artifact(artifact, location, name, object_store, save_fn)
    os.remove(os.path.join(location, name))


def save_artifact(
    artifact: T, location: str, name: str, object_store: Optional[ObjectStore], save_fn: Callable[[T, str], None]
) -> None:
    # Should be run on rank zero only!
    # Ensure local directory exists
    os.makedirs(location, exist_ok=True)
    # Save artifact locally
    artifact_name = os.path.join(location, name)
    save_fn(artifact, artifact_name)
    # If object store is provided, upload artifact to object store
    if object_store is not None:
        object_store.upload_object(artifact_name, artifact_name)


def load_object(location: str, name: str, object_store: ObjectStore, load_fn: Callable[[str], T]) -> T:
    artifact_name = os.path.join(location, name)
    stream = object_store.download_object_as_stream(artifact_name)
    artifact = load_fn(io.BytesIO(b"".join(stream)))
    return artifact


def load_artifact(location: str, name: str, object_store: Optional[ObjectStore], load_fn: Callable[[str], T]) -> T:
    # Should run on all ranks
    # Path to artifact
    artifact_name = os.path.join(location, name)
    # Ensure local directory exists and download object if object store, rank zero only!
    if dist.get_global_rank() == 0:
        os.makedirs(location, exist_ok=True)
        if object_exists(artifact_name, object_store):
            object_store.download_object(artifact_name, artifact_name, overwrite_existing=True)
    dist.barrier()
    # Load from all ranks
    artifact = load_fn(artifact_name)
    return artifact


def MSE(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "sum",
) -> torch.Tensor:
    target_type = infer_target_type(input, target)
    if target_type == "indices":
        return 0.5 * ((F.one_hot(target).float() - input) ** 2).sum(dim=-1).mean()
        # return F.mse_loss(input, F.one_hot(target).float(), size_average, reduce, reduction)
    elif target_type == "one_hot":
        return 0.5 * ((input - target) ** 2).sum(dim=-1).mean()
        # return F.mse_loss(input, target, size_average, reduce, reduction)
    else:
        raise ValueError(f"Unrecognized target type {target_type}")
