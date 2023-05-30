# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from sgdcollapse.models.ComposerClassifier import ComposerClassifier
from composer.models.model_hparams import ModelHparams


import yahp as hp

# __all__ = ["ComposerDeepLinear"]


class VGG16_NET(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16_NET, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc14 = nn.Linear(512, 512)
        self.fc15 = nn.Linear(512, 512)
        self.fc16 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.maxpool(x)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.maxpool(x)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.maxpool(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc14(x))
        x = F.relu(self.fc15(x))
        x = self.fc16(x)
        return x


class VGG16_NET_GELU(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16_NET_GELU, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc14 = nn.Linear(512, 512)
        self.fc15 = nn.Linear(512, 512)
        self.fc16 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = self.maxpool(x)
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        x = self.maxpool(x)
        x = F.gelu(self.conv5(x))
        x = F.gelu(self.conv6(x))
        x = F.gelu(self.conv7(x))
        x = self.maxpool(x)
        x = F.gelu(self.conv8(x))
        x = F.gelu(self.conv9(x))
        x = F.gelu(self.conv10(x))
        x = self.maxpool(x)
        x = F.gelu(self.conv11(x))
        x = F.gelu(self.conv12(x))
        x = F.gelu(self.conv13(x))
        x = self.maxpool(x)
        x = x.reshape(x.size(0), -1)
        x = F.gelu(self.fc14(x))
        x = F.gelu(self.fc15(x))
        x = self.fc16(x)
        return x


class VGG16_NET_GELU_DO(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16_NET_GELU_DO, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc14 = nn.Linear(512, 512)
        self.fc15 = nn.Linear(512, 512)
        self.fc16 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = self.maxpool(x)
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        x = self.maxpool(x)
        x = F.gelu(self.conv5(x))
        x = F.gelu(self.conv6(x))
        x = F.gelu(self.conv7(x))
        x = self.maxpool(x)
        x = F.gelu(self.conv8(x))
        x = F.gelu(self.conv9(x))
        x = F.gelu(self.conv10(x))
        x = self.maxpool(x)
        x = F.gelu(self.conv11(x))
        x = F.gelu(self.conv12(x))
        x = F.gelu(self.conv13(x))
        x = self.maxpool(x)
        x = x.reshape(x.size(0), -1)
        x = F.gelu(self.fc14(x))
        x = F.dropout(x, 0.5)
        x = F.gelu(self.fc15(x))
        x = F.dropout(x, 0.5)
        x = self.fc16(x)
        return x


class VGG16_NET_BN_GELU(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16_NET_BN_GELU, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.GELU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.GELU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.GELU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.GELU()
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.GELU()
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.GELU()
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.GELU()
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.GELU()
        )
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(nn.Linear(512, 512), nn.GELU())
        self.fc1 = nn.Sequential(nn.Linear(512, 512), nn.GELU())
        self.fc2 = nn.Sequential(nn.Linear(512, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class ComposerVGG16(ComposerClassifier):
    def __init__(self, loss_name: str = "mse", num_classes=10) -> None:
        model = VGG16_NET(num_classes)

        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)  # xavier_uniform(m.weight)
                m.bias.data.fill_(0.0)

        model.apply(init_weights)
        super().__init__(module=model, loss_name=loss_name)


class ComposerVGG16_GELU(ComposerClassifier):
    def __init__(self, loss_name: str = "mse", num_classes=10) -> None:
        model = VGG16_NET_GELU(num_classes)

        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)  # xavier_uniform(m.weight)
                m.bias.data.fill_(0.0)

        model.apply(init_weights)
        super().__init__(module=model, loss_name=loss_name)


class ComposerVGG16_GELU_DO(ComposerClassifier):
    def __init__(self, loss_name: str = "mse", num_classes=10) -> None:
        model = VGG16_NET_GELU_DO(num_classes)

        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)  # xavier_uniform(m.weight)
                m.bias.data.fill_(0.0)

        model.apply(init_weights)
        super().__init__(module=model, loss_name=loss_name)


class ComposerVGG16_BN_GELU(ComposerClassifier):
    def __init__(self, loss_name: str = "mse", num_classes=10) -> None:
        model = VGG16_NET_BN_GELU(num_classes)

        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)  # xavier_uniform(m.weight)
                m.bias.data.fill_(0.0)

        model.apply(init_weights)
        super().__init__(module=model, loss_name=loss_name)


@dataclass
class ComposerVGG16Hparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.ComposerResNet`.

    Args:
        model_name (str): Name of the ResNet model instance. Either [``"resnet18"``, ``"resnet34"``, ``"resnet50"``, ``"resnet101"``,
            ``"resnet152"``].
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``1000``.
        pretrained (bool, optional): If True, use ImageNet pretrained weights. Default: ``False``.
        groups (int, optional): Number of filter groups for the 3x3 convolution layer in bottleneck blocks. Default: ``1``.
        width_per_group (int, optional): Initial width for each convolution group. Width doubles after each stage.
            Default: ``64``.
        initializers (List[Initializer], optional): Initializers for the model. ``None`` for no initialization.
            Default: ``None``.
    """

    loss_name: str = hp.optional(
        "Name of loss function. E.g. 'soft_cross_entropy' or 'binary_cross_entropy_with_logits'. (default: ``soft_cross_entropy``)",
        default="mse",
    )
    num_classes: int = hp.optional(
        "",
        default=10,
    )

    def validate(self):
        pass

    def initialize_object(self):
        return ComposerVGG16(
            loss_name=self.loss_name,
            num_classes=self.num_classes,
        )


@dataclass
class ComposerVGG16GELUHparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.ComposerResNet`.

    Args:
        model_name (str): Name of the ResNet model instance. Either [``"resnet18"``, ``"resnet34"``, ``"resnet50"``, ``"resnet101"``,
            ``"resnet152"``].
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``1000``.
        pretrained (bool, optional): If True, use ImageNet pretrained weights. Default: ``False``.
        groups (int, optional): Number of filter groups for the 3x3 convolution layer in bottleneck blocks. Default: ``1``.
        width_per_group (int, optional): Initial width for each convolution group. Width doubles after each stage.
            Default: ``64``.
        initializers (List[Initializer], optional): Initializers for the model. ``None`` for no initialization.
            Default: ``None``.
    """

    loss_name: str = hp.optional(
        "Name of loss function. E.g. 'soft_cross_entropy' or 'binary_cross_entropy_with_logits'. (default: ``soft_cross_entropy``)",
        default="mse",
    )

    num_classes: int = hp.optional(
        "",
        default=10,
    )

    def validate(self):
        pass

    def initialize_object(self):
        return ComposerVGG16_GELU(
            loss_name=self.loss_name,
            num_classes=self.num_classes,
        )


@dataclass
class ComposerVGG16GELUDOHparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.ComposerResNet`.

    Args:
        model_name (str): Name of the ResNet model instance. Either [``"resnet18"``, ``"resnet34"``, ``"resnet50"``, ``"resnet101"``,
            ``"resnet152"``].
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``1000``.
        pretrained (bool, optional): If True, use ImageNet pretrained weights. Default: ``False``.
        groups (int, optional): Number of filter groups for the 3x3 convolution layer in bottleneck blocks. Default: ``1``.
        width_per_group (int, optional): Initial width for each convolution group. Width doubles after each stage.
            Default: ``64``.
        initializers (List[Initializer], optional): Initializers for the model. ``None`` for no initialization.
            Default: ``None``.
    """

    loss_name: str = hp.optional(
        "Name of loss function. E.g. 'soft_cross_entropy' or 'binary_cross_entropy_with_logits'. (default: ``soft_cross_entropy``)",
        default="mse",
    )

    num_classes: int = hp.optional(
        "",
        default=10,
    )

    def validate(self):
        pass

    def initialize_object(self):
        return ComposerVGG16_GELU_DO(
            loss_name=self.loss_name,
            num_classes=self.num_classes,
        )


@dataclass
class ComposerVGG16BNGELUHparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.ComposerResNet`.

    Args:
        model_name (str): Name of the ResNet model instance. Either [``"resnet18"``, ``"resnet34"``, ``"resnet50"``, ``"resnet101"``,
            ``"resnet152"``].
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``1000``.
        pretrained (bool, optional): If True, use ImageNet pretrained weights. Default: ``False``.
        groups (int, optional): Number of filter groups for the 3x3 convolution layer in bottleneck blocks. Default: ``1``.
        width_per_group (int, optional): Initial width for each convolution group. Width doubles after each stage.
            Default: ``64``.
        initializers (List[Initializer], optional): Initializers for the model. ``None`` for no initialization.
            Default: ``None``.
    """

    loss_name: str = hp.optional(
        "Name of loss function. E.g. 'soft_cross_entropy' or 'binary_cross_entropy_with_logits'. (default: ``soft_cross_entropy``)",
        default="mse",
    )

    num_classes: int = hp.optional(
        "",
        default=10,
    )

    def validate(self):
        pass

    def initialize_object(self):
        return ComposerVGG16_BN_GELU(
            loss_name=self.loss_name,
            number_classes=self.num_classes,
        )
