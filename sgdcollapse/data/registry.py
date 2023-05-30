from typing import Type

import composer.datasets as composer_datasets

from sgdcollapse import utils
from sgdcollapse.data.cifar100_dataset_hparams import CIFAR100DatasetHparams
from sgdcollapse.data.imagenet import ImagenetDatasetHparams
from sgdcollapse.data.cifar import (
    CIFAR10DatasetHparams,
    CIFAR10NoisyDatasetHparams,
    CIFAR10NoInputNoiseDatasetHparams,
    CIFAR10LabelNoiseDatasetHparams,
)
from sgdcollapse.data.cifar100_dataset_hparams import CIFAR100LabelNoiseDatasetHparams

data_registry: utils.registry.Registry[str, Type[composer_datasets.DatasetHparams]] = {
    "cifar10": CIFAR10DatasetHparams,
    "cifar100": CIFAR100DatasetHparams,
    "imagenet": ImagenetDatasetHparams,
    "cifar10noisy": CIFAR10NoisyDatasetHparams,
    "cifar10noiseless": CIFAR10NoInputNoiseDatasetHparams,
    "cifar10labelnoise": CIFAR10LabelNoiseDatasetHparams,
    "cifar100labelnoise": CIFAR100LabelNoiseDatasetHparams,
}
