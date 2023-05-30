from typing import Type

import composer.models as composer_models

from sgdcollapse import utils
from sgdcollapse.models.Toymodel import (
    ComposerVGG16Hparams,
    ComposerVGG16GELUHparams,
    ComposerVGG16BNGELUHparams,
    ComposerVGG16GELUDOHparams,
)
from sgdcollapse.models.ResNet import ResNetLowresHparams

model_registry: utils.registry.Registry[str, Type[composer_models.ModelHparams]] = {
    "resnet": composer_models.ResNetHparams,
    "vgg16": ComposerVGG16Hparams,
    "vgg16gelu": ComposerVGG16GELUHparams,
    "vgg16bngelu": ComposerVGG16BNGELUHparams,
    "vgg16geludo": ComposerVGG16GELUDOHparams,
    "resnetlowres": ResNetLowresHparams,
}
