from dataclasses import dataclass

from composer.models import ComposerResNet, ResNetHparams
from torch import nn


@dataclass
class ResNetLowresHparams(ResNetHparams):
    def initialize_object(self) -> ComposerResNet:
        assert self.model_name == "resnet18"
        model = super().initialize_object()
        model.module.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.module.maxpool = nn.Identity()
        model.module.relu = nn.GELU()
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            getattr(model.module, layer_name)[0].relu = nn.GELU()
            getattr(model.module, layer_name)[1].relu = nn.GELU()
        return model
