import os
from dataclasses import dataclass

from composer.datasets import DataLoaderHparams, DatasetHparams
from composer.utils import dist
from torchvision import transforms
from torchvision.datasets import CIFAR100
import torch
from typing import List

import yahp as hp


@dataclass
class CIFAR100LabelNoiseDatasetHparams(DatasetHparams):
    download: bool = hp.optional("whether to download the dataset, if needed", default=True)
    sigma: float = hp.optional("sigma for gaussian noise", default=0.0)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams, indices: List[int] = None):
        datadir = self.datadir
        if datadir is None:
            datadir = os.environ["CIFAR100_DIR"]

        cifar100_mean = 0.5071, 0.4867, 0.4408
        cifar100_std = 0.2675, 0.2565, 0.2761

        if self.is_train:
            transformation = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar100_mean, cifar100_std),
                ]
            )

            def label_noise(z):
                def sub_func(x):
                    if torch.rand(1) < z:
                        # label=torch.zeros(x.size(-1))
                        # label[torch.randint(0,10,(1,))]=1
                        while True:
                            new_label = torch.randint(0, 100, (1,)).ravel()
                            if new_label != x.ravel():
                                return new_label
                    else:
                        return x.ravel()

                return sub_func

            transformation_target = transforms.Compose(
                [
                    lambda x: torch.LongTensor([x]),  # or just torch.tensor
                    # lambda x:torch.nn.functional.one_hot(x,10).ravel(),
                    label_noise(self.sigma),
                    lambda x: torch.nn.functional.one_hot(x, 100).ravel(),
                ]
            )

        else:
            transformation = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(cifar100_mean, cifar100_std),
                ]
            )

        dataset = CIFAR100(
            datadir,
            train=self.is_train,
            download=False,
            transform=transformation,
            target_transform=transformation_target,
        )

        if indices is not None:
            dataset = torch.utils.data.Subset(dataset, indices)

        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)

        return dataloader_hparams.initialize_object(
            dataset, batch_size=batch_size, sampler=sampler, drop_last=self.drop_last
        )


@dataclass
class CIFAR100DatasetHparams(DatasetHparams):
    download: bool = hp.optional("whether to download the dataset, if needed", default=True)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams, indices: List[int] = None):
        datadir = self.datadir
        if datadir is None:
            datadir = os.environ["CIFAR100_DIR"]

        cifar100_mean = 0.5071, 0.4867, 0.4408
        cifar100_std = 0.2675, 0.2565, 0.2761

        if self.is_train:
            transformation = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar100_mean, cifar100_std),
                ]
            )
        else:
            transformation = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(cifar100_mean, cifar100_std),
                ]
            )

        dataset = CIFAR100(
            datadir,
            train=self.is_train,
            download=False,
            transform=transformation,
        )

        if indices is not None:
            dataset = torch.utils.data.Subset(dataset, indices)

        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)

        return dataloader_hparams.initialize_object(
            dataset, batch_size=batch_size, sampler=sampler, drop_last=self.drop_last
        )
