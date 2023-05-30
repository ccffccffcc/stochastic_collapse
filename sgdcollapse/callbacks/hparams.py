# TODO: Docstring
import os
from dataclasses import dataclass
from typing import Optional

import yahp as hp
from composer.callbacks import CallbackHparams, CheckpointSaver
from composer.core.time import Time
from composer.utils.import_helpers import import_object


@dataclass
class CheckpointSaverHparams(CallbackHparams):
    save_folder: str = hp.optional(
        doc="Location to save checkpoints", default=f"{os.environ['EXP_DIR']}/" + "{run_name}/checkpoints"
    )
    filename: str = hp.optional("File name format", default="state_{batch}ba.pt")
    artifact_name: str = hp.optional(
        "Artifact name format", default=f"{os.environ['EXP_DIR']}/" + "{run_name}/checkpoints/state_{batch}ba.pt"
    )
    latest_filename: Optional[str] = hp.optional("Latest checkpoint symlink format", default=None)
    latest_artifact_name: Optional[str] = hp.optional("Latest artifact symlink format", default=None)
    save_interval: str = hp.optional("Save interval time or whether to save: `(State, Event) -> bool`", default="1ep")
    overwrite: bool = hp.optional("Whether to override existing checkpoints.", default=True)
    num_checkpoints_to_keep: int = hp.optional("Number checkpoints to persist locally, keep all: -1", default=-1)
    weights_only: bool = hp.optional("Whether to save only model weights.", default=False)

    def initialize_object(self) -> CheckpointSaver:
        try:
            save_interval = Time.from_timestring(self.save_interval)
        except ValueError:
            # assume it is a function path
            save_interval = import_object(self.save_interval)
        return CheckpointSaver(
            folder=self.save_folder,
            filename=self.filename,
            artifact_name=self.artifact_name,
            latest_filename=self.latest_filename,
            latest_artifact_name=self.latest_artifact_name,
            save_interval=save_interval,
            overwrite=self.overwrite,
            num_checkpoints_to_keep=self.num_checkpoints_to_keep,
            weights_only=self.weights_only,
        )
