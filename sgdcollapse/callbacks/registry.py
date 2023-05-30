from typing import Type

import composer.callbacks as composer_callbacks

from sgdcollapse import utils
from sgdcollapse.callbacks.hparams import CheckpointSaverHparams

callback_registry: utils.registry.Registry[str, Type[composer_callbacks.CallbackHparams]] = {
    "checkpoint_saver": CheckpointSaverHparams,
    "lr_monitor": composer_callbacks.LRMonitorHparams,
    "grad_monitor": composer_callbacks.GradMonitorHparams,
    "speed_monitor": composer_callbacks.SpeedMonitorHparams,
}
