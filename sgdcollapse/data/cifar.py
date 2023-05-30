# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""CIFAR image classification dataset.

The CIFAR datasets are a collection of labeled 32x32 colour images. Please refer to the `CIFAR dataset
<https://www.cs.toronto.edu/~kriz/cifar.html>`_ for more details.
"""

import logging
import os
import textwrap
from dataclasses import dataclass
from typing import List
import numpy as np

import torch
import yahp as hp
from torchvision import transforms
from torchvision.datasets import CIFAR10

from composer.datasets.dataloader import DataLoaderHparams
from composer.datasets.ffcv_utils import write_ffcv_dataset
from composer.datasets.hparams import DatasetHparams, SyntheticHparamsMixin, WebDatasetHparams
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.utils import dist

__all__ = [
    "CIFAR10DatasetHparams",
    "CIFARWebDatasetHparams",
    "CIFAR10WebDatasetHparams",
    "CIFAR20WebDatasetHparams",
    "CIFAR100WebDatasetHparams",
]

log = logging.getLogger(__name__)


@dataclass
class CIFAR10DatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the CIFAR-10 dataset for image classification from a local disk.

    Args:
        download (bool): Whether to download the dataset, if needed. Default: ``True``.
        use_ffcv (bool): Whether to use FFCV dataloaders. Default: ``False``.
        ffcv_dir (str): A directory containing train/val <file>.ffcv files. If these files don't exist and
            ``ffcv_write_dataset`` is ``True``, train/val <file>.ffcv files will be created in this dir. Default: ``"/tmp"``.
        ffcv_dest (str): <file>.ffcv file that has dataset samples. Default: ``"cifar_train.ffcv"``.
        ffcv_write_dataset (std): Whether to create dataset in FFCV format (<file>.ffcv) if it doesn't exist. Default:
        ``False``.
    """

    download: bool = hp.optional("whether to download the dataset, if needed", default=True)
    use_ffcv: bool = hp.optional("whether to use ffcv for faster dataloading", default=False)
    ffcv_dir: str = hp.optional(
        "A directory containing train/val <file>.ffcv files. If these files don't exist and ffcv_write_dataset is true, train/val <file>.ffcv files will be created in this dir.",
        default="/tmp",
    )
    ffcv_dest: str = hp.optional("<file>.ffcv file that has dataset samples", default="cifar_train.ffcv")
    ffcv_write_dataset: bool = hp.optional(
        "Whether to create dataset in FFCV format (<file>.ffcv) if it doesn't exist", default=False
    )

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams, indices: List[int] = None):
        if self.use_synthetic:
            total_dataset_size = 50_000 if self.is_train else 10_000
            dataset = SyntheticBatchPairDataset(
                total_dataset_size=total_dataset_size,
                data_shape=[3, 32, 32],
                num_classes=10,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )

        elif self.use_ffcv:
            try:
                import ffcv  # type: ignore
                from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder  # type: ignore
                from ffcv.pipeline.operation import Operation  # type: ignore
            except ImportError:
                raise ImportError(
                    textwrap.dedent(
                        """\
                    Composer was installed without ffcv support.
                    To use ffcv with Composer, please install ffcv in your environment."""
                    )
                )

            dataset_filepath = os.path.join(self.ffcv_dir, self.ffcv_dest)
            # always create if ffcv_write_dataset is true
            if self.ffcv_write_dataset:
                if dist.get_local_rank() == 0:
                    if self.datadir is None:
                        raise ValueError(
                            "datadir is required if use_synthetic is False and ffcv_write_dataset is True."
                        )
                    ds = CIFAR10(
                        self.datadir,
                        train=self.is_train,
                        download=self.download,
                    )

                    write_ffcv_dataset(dataset=ds, write_path=dataset_filepath)

                # Wait for the local rank 0 to be done creating the dataset in ffcv format.
                dist.barrier()

            if not os.path.exists(dataset_filepath):
                raise ValueError(
                    f"Dataset file containing samples not found at {dataset_filepath}. Use ffcv_dir flag to point to a dir containing {dataset_filepath}."
                )

            # Please note that this mean/std is different from the mean/std used for regular PyTorch dataloader as
            # ToTensor does the normalization for PyTorch dataloaders.
            cifar10_mean_ffcv = [125.307, 122.961, 113.8575]
            cifar10_std_ffcv = [51.5865, 50.847, 51.255]
            label_pipeline: List[Operation] = [IntDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.Squeeze()]
            image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

            if self.is_train:
                image_pipeline.extend(
                    [
                        ffcv.transforms.RandomHorizontalFlip(),
                        ffcv.transforms.RandomTranslate(padding=2, fill=tuple(map(int, cifar10_mean_ffcv))),
                        ffcv.transforms.Cutout(4, tuple(map(int, cifar10_mean_ffcv))),
                    ]
                )
            # Common transforms for train and test
            image_pipeline.extend(
                [
                    ffcv.transforms.ToTensor(),
                    ffcv.transforms.ToTorchImage(channels_last=False, convert_back_int16=False),
                    ffcv.transforms.Convert(torch.float32),
                    transforms.Normalize(cifar10_mean_ffcv, cifar10_std_ffcv),
                ]
            )

            ordering = ffcv.loader.OrderOption.RANDOM if self.is_train else ffcv.loader.OrderOption.SEQUENTIAL

            return ffcv.Loader(
                dataset_filepath,
                batch_size=batch_size,
                num_workers=dataloader_hparams.num_workers,
                order=ordering,
                distributed=False,
                pipelines={"image": image_pipeline, "label": label_pipeline},
                batches_ahead=dataloader_hparams.prefetch_factor,
                drop_last=self.drop_last,
            )
        else:
            if self.datadir is None:
                raise ValueError("datadir is required if use_synthetic is False")

            cifar10_mean = 0.4914, 0.4822, 0.4465
            cifar10_std = 0.247, 0.243, 0.261

            if self.is_train:
                transformation = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(cifar10_mean, cifar10_std),
                    ]
                )
            else:
                transformation = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(cifar10_mean, cifar10_std),
                    ]
                )

            dataset = CIFAR10(
                self.datadir,
                train=self.is_train,
                download=self.download,
                transform=transformation,
            )

        if indices is not None:
            dataset = torch.utils.data.Subset(dataset, indices)

        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)

        return dataloader_hparams.initialize_object(
            dataset, batch_size=batch_size, sampler=sampler, drop_last=self.drop_last
        )


@dataclass
class CIFAR10NoInputNoiseDatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the CIFAR-10 dataset for image classification from a local disk.

    Args:
        download (bool): Whether to download the dataset, if needed. Default: ``True``.
        use_ffcv (bool): Whether to use FFCV dataloaders. Default: ``False``.
        ffcv_dir (str): A directory containing train/val <file>.ffcv files. If these files don't exist and
            ``ffcv_write_dataset`` is ``True``, train/val <file>.ffcv files will be created in this dir. Default: ``"/tmp"``.
        ffcv_dest (str): <file>.ffcv file that has dataset samples. Default: ``"cifar_train.ffcv"``.
        ffcv_write_dataset (std): Whether to create dataset in FFCV format (<file>.ffcv) if it doesn't exist. Default:
        ``False``.
    """

    download: bool = hp.optional("whether to download the dataset, if needed", default=True)
    use_ffcv: bool = hp.optional("whether to use ffcv for faster dataloading", default=False)
    ffcv_dir: str = hp.optional(
        "A directory containing train/val <file>.ffcv files. If these files don't exist and ffcv_write_dataset is true, train/val <file>.ffcv files will be created in this dir.",
        default="/tmp",
    )
    ffcv_dest: str = hp.optional("<file>.ffcv file that has dataset samples", default="cifar_train.ffcv")
    ffcv_write_dataset: bool = hp.optional(
        "Whether to create dataset in FFCV format (<file>.ffcv) if it doesn't exist", default=False
    )

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams, indices: List[int] = None):
        if self.use_synthetic:
            total_dataset_size = 50_000 if self.is_train else 10_000
            dataset = SyntheticBatchPairDataset(
                total_dataset_size=total_dataset_size,
                data_shape=[3, 32, 32],
                num_classes=10,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )

        elif self.use_ffcv:
            try:
                import ffcv  # type: ignore
                from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder  # type: ignore
                from ffcv.pipeline.operation import Operation  # type: ignore
            except ImportError:
                raise ImportError(
                    textwrap.dedent(
                        """\
                    Composer was installed without ffcv support.
                    To use ffcv with Composer, please install ffcv in your environment."""
                    )
                )

            dataset_filepath = os.path.join(self.ffcv_dir, self.ffcv_dest)
            # always create if ffcv_write_dataset is true
            if self.ffcv_write_dataset:
                if dist.get_local_rank() == 0:
                    if self.datadir is None:
                        raise ValueError(
                            "datadir is required if use_synthetic is False and ffcv_write_dataset is True."
                        )
                    ds = CIFAR10(
                        self.datadir,
                        train=self.is_train,
                        download=self.download,
                    )

                    write_ffcv_dataset(dataset=ds, write_path=dataset_filepath)

                # Wait for the local rank 0 to be done creating the dataset in ffcv format.
                dist.barrier()

            if not os.path.exists(dataset_filepath):
                raise ValueError(
                    f"Dataset file containing samples not found at {dataset_filepath}. Use ffcv_dir flag to point to a dir containing {dataset_filepath}."
                )

            # Please note that this mean/std is different from the mean/std used for regular PyTorch dataloader as
            # ToTensor does the normalization for PyTorch dataloaders.
            cifar10_mean_ffcv = [125.307, 122.961, 113.8575]
            cifar10_std_ffcv = [51.5865, 50.847, 51.255]
            label_pipeline: List[Operation] = [IntDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.Squeeze()]
            image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

            if self.is_train:
                image_pipeline.extend(
                    [
                        ffcv.transforms.RandomHorizontalFlip(),
                        ffcv.transforms.RandomTranslate(padding=2, fill=tuple(map(int, cifar10_mean_ffcv))),
                        ffcv.transforms.Cutout(4, tuple(map(int, cifar10_mean_ffcv))),
                    ]
                )
            # Common transforms for train and test
            image_pipeline.extend(
                [
                    ffcv.transforms.ToTensor(),
                    ffcv.transforms.ToTorchImage(channels_last=False, convert_back_int16=False),
                    ffcv.transforms.Convert(torch.float32),
                    transforms.Normalize(cifar10_mean_ffcv, cifar10_std_ffcv),
                ]
            )

            ordering = ffcv.loader.OrderOption.RANDOM if self.is_train else ffcv.loader.OrderOption.SEQUENTIAL

            return ffcv.Loader(
                dataset_filepath,
                batch_size=batch_size,
                num_workers=dataloader_hparams.num_workers,
                order=ordering,
                distributed=False,
                pipelines={"image": image_pipeline, "label": label_pipeline},
                batches_ahead=dataloader_hparams.prefetch_factor,
                drop_last=self.drop_last,
            )
        else:
            if self.datadir is None:
                raise ValueError("datadir is required if use_synthetic is False")

            cifar10_mean = 0.4914, 0.4822, 0.4465
            cifar10_std = 0.247, 0.243, 0.261

            if self.is_train:
                transformation = transforms.Compose(
                    [
                        # transforms.RandomCrop(32, padding=4),
                        # transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(cifar10_mean, cifar10_std),
                    ]
                )
            else:
                transformation = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(cifar10_mean, cifar10_std),
                    ]
                )

            dataset = CIFAR10(
                self.datadir,
                train=self.is_train,
                download=self.download,
                transform=transformation,
            )

        if indices is not None:
            dataset = torch.utils.data.Subset(dataset, indices)

        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)

        return dataloader_hparams.initialize_object(
            dataset, batch_size=batch_size, sampler=sampler, drop_last=self.drop_last
        )


@dataclass
class CIFAR10NoisyDatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the CIFAR-10 dataset for image classification from a local disk.

    Args:
        download (bool): Whether to download the dataset, if needed. Default: ``True``.
        use_ffcv (bool): Whether to use FFCV dataloaders. Default: ``False``.
        ffcv_dir (str): A directory containing train/val <file>.ffcv files. If these files don't exist and
            ``ffcv_write_dataset`` is ``True``, train/val <file>.ffcv files will be created in this dir. Default: ``"/tmp"``.
        ffcv_dest (str): <file>.ffcv file that has dataset samples. Default: ``"cifar_train.ffcv"``.
        ffcv_write_dataset (std): Whether to create dataset in FFCV format (<file>.ffcv) if it doesn't exist. Default:
        ``False``.
    """

    download: bool = hp.optional("whether to download the dataset, if needed", default=True)
    use_ffcv: bool = hp.optional("whether to use ffcv for faster dataloading", default=False)
    sigma: float = hp.optional("sigma for gaussian noise", default=0.0)
    ffcv_dir: str = hp.optional(
        "A directory containing train/val <file>.ffcv files. If these files don't exist and ffcv_write_dataset is true, train/val <file>.ffcv files will be created in this dir.",
        default="/tmp",
    )
    ffcv_dest: str = hp.optional("<file>.ffcv file that has dataset samples", default="cifar_train.ffcv")
    ffcv_write_dataset: bool = hp.optional(
        "Whether to create dataset in FFCV format (<file>.ffcv) if it doesn't exist", default=False
    )

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams, indices: List[int] = None):
        if self.use_synthetic:
            total_dataset_size = 50_000 if self.is_train else 10_000
            dataset = SyntheticBatchPairDataset(
                total_dataset_size=total_dataset_size,
                data_shape=[3, 32, 32],
                num_classes=10,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )

        elif self.use_ffcv:
            try:
                import ffcv  # type: ignore
                from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder  # type: ignore
                from ffcv.pipeline.operation import Operation  # type: ignore
            except ImportError:
                raise ImportError(
                    textwrap.dedent(
                        """\
                    Composer was installed without ffcv support.
                    To use ffcv with Composer, please install ffcv in your environment."""
                    )
                )

            dataset_filepath = os.path.join(self.ffcv_dir, self.ffcv_dest)
            # always create if ffcv_write_dataset is true
            if self.ffcv_write_dataset:
                if dist.get_local_rank() == 0:
                    if self.datadir is None:
                        raise ValueError(
                            "datadir is required if use_synthetic is False and ffcv_write_dataset is True."
                        )
                    ds = CIFAR10(
                        self.datadir,
                        train=self.is_train,
                        download=self.download,
                    )

                    write_ffcv_dataset(dataset=ds, write_path=dataset_filepath)

                # Wait for the local rank 0 to be done creating the dataset in ffcv format.
                dist.barrier()

            if not os.path.exists(dataset_filepath):
                raise ValueError(
                    f"Dataset file containing samples not found at {dataset_filepath}. Use ffcv_dir flag to point to a dir containing {dataset_filepath}."
                )

            # Please note that this mean/std is different from the mean/std used for regular PyTorch dataloader as
            # ToTensor does the normalization for PyTorch dataloaders.
            cifar10_mean_ffcv = [125.307, 122.961, 113.8575]
            cifar10_std_ffcv = [51.5865, 50.847, 51.255]
            label_pipeline: List[Operation] = [IntDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.Squeeze()]
            image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

            if self.is_train:
                image_pipeline.extend(
                    [
                        ffcv.transforms.RandomHorizontalFlip(),
                        ffcv.transforms.RandomTranslate(padding=2, fill=tuple(map(int, cifar10_mean_ffcv))),
                        ffcv.transforms.Cutout(4, tuple(map(int, cifar10_mean_ffcv))),
                    ]
                )
            # Common transforms for train and test
            image_pipeline.extend(
                [
                    ffcv.transforms.ToTensor(),
                    ffcv.transforms.ToTorchImage(channels_last=False, convert_back_int16=False),
                    ffcv.transforms.Convert(torch.float32),
                    transforms.Normalize(cifar10_mean_ffcv, cifar10_std_ffcv),
                ]
            )

            ordering = ffcv.loader.OrderOption.RANDOM if self.is_train else ffcv.loader.OrderOption.SEQUENTIAL

            return ffcv.Loader(
                dataset_filepath,
                batch_size=batch_size,
                num_workers=dataloader_hparams.num_workers,
                order=ordering,
                distributed=False,
                pipelines={"image": image_pipeline, "label": label_pipeline},
                batches_ahead=dataloader_hparams.prefetch_factor,
                drop_last=self.drop_last,
            )
        else:
            if self.datadir is None:
                raise ValueError("datadir is required if use_synthetic is False")

            cifar10_mean = 0.4914, 0.4822, 0.4465
            cifar10_std = 0.247, 0.243, 0.261

            if self.is_train:
                transformation = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(cifar10_mean, cifar10_std),
                    ]
                )

                transformation_target = transforms.Compose(
                    [
                        lambda x: torch.LongTensor([x]),  # or just torch.tensor
                        lambda x: torch.nn.functional.one_hot(x, 10).ravel(),
                        lambda x: x + self.sigma * torch.randn(10),
                    ]
                )
            else:
                transformation = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(cifar10_mean, cifar10_std),
                    ]
                )

            dataset = CIFAR10(
                self.datadir,
                train=self.is_train,
                download=self.download,
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
class CIFAR10LabelNoiseDatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the CIFAR-10 dataset for image classification from a local disk.

    Args:
        download (bool): Whether to download the dataset, if needed. Default: ``True``.
        use_ffcv (bool): Whether to use FFCV dataloaders. Default: ``False``.
        ffcv_dir (str): A directory containing train/val <file>.ffcv files. If these files don't exist and
            ``ffcv_write_dataset`` is ``True``, train/val <file>.ffcv files will be created in this dir. Default: ``"/tmp"``.
        ffcv_dest (str): <file>.ffcv file that has dataset samples. Default: ``"cifar_train.ffcv"``.
        ffcv_write_dataset (std): Whether to create dataset in FFCV format (<file>.ffcv) if it doesn't exist. Default:
        ``False``.
    """

    download: bool = hp.optional("whether to download the dataset, if needed", default=True)
    use_ffcv: bool = hp.optional("whether to use ffcv for faster dataloading", default=False)
    sigma: float = hp.optional("sigma for gaussian noise", default=0.0)
    ffcv_dir: str = hp.optional(
        "A directory containing train/val <file>.ffcv files. If these files don't exist and ffcv_write_dataset is true, train/val <file>.ffcv files will be created in this dir.",
        default="/tmp",
    )
    ffcv_dest: str = hp.optional("<file>.ffcv file that has dataset samples", default="cifar_train.ffcv")
    ffcv_write_dataset: bool = hp.optional(
        "Whether to create dataset in FFCV format (<file>.ffcv) if it doesn't exist", default=False
    )

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams, indices: List[int] = None):
        if self.use_synthetic:
            total_dataset_size = 50_000 if self.is_train else 10_000
            dataset = SyntheticBatchPairDataset(
                total_dataset_size=total_dataset_size,
                data_shape=[3, 32, 32],
                num_classes=10,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )

        elif self.use_ffcv:
            try:
                import ffcv  # type: ignore
                from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder  # type: ignore
                from ffcv.pipeline.operation import Operation  # type: ignore
            except ImportError:
                raise ImportError(
                    textwrap.dedent(
                        """\
                    Composer was installed without ffcv support.
                    To use ffcv with Composer, please install ffcv in your environment."""
                    )
                )

            dataset_filepath = os.path.join(self.ffcv_dir, self.ffcv_dest)
            # always create if ffcv_write_dataset is true
            if self.ffcv_write_dataset:
                if dist.get_local_rank() == 0:
                    if self.datadir is None:
                        raise ValueError(
                            "datadir is required if use_synthetic is False and ffcv_write_dataset is True."
                        )
                    ds = CIFAR10(
                        self.datadir,
                        train=self.is_train,
                        download=self.download,
                    )

                    write_ffcv_dataset(dataset=ds, write_path=dataset_filepath)

                # Wait for the local rank 0 to be done creating the dataset in ffcv format.
                dist.barrier()

            if not os.path.exists(dataset_filepath):
                raise ValueError(
                    f"Dataset file containing samples not found at {dataset_filepath}. Use ffcv_dir flag to point to a dir containing {dataset_filepath}."
                )

            # Please note that this mean/std is different from the mean/std used for regular PyTorch dataloader as
            # ToTensor does the normalization for PyTorch dataloaders.
            cifar10_mean_ffcv = [125.307, 122.961, 113.8575]
            cifar10_std_ffcv = [51.5865, 50.847, 51.255]
            label_pipeline: List[Operation] = [IntDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.Squeeze()]
            image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

            if self.is_train:
                image_pipeline.extend(
                    [
                        ffcv.transforms.RandomHorizontalFlip(),
                        ffcv.transforms.RandomTranslate(padding=2, fill=tuple(map(int, cifar10_mean_ffcv))),
                        ffcv.transforms.Cutout(4, tuple(map(int, cifar10_mean_ffcv))),
                    ]
                )
            # Common transforms for train and test
            image_pipeline.extend(
                [
                    ffcv.transforms.ToTensor(),
                    ffcv.transforms.ToTorchImage(channels_last=False, convert_back_int16=False),
                    ffcv.transforms.Convert(torch.float32),
                    transforms.Normalize(cifar10_mean_ffcv, cifar10_std_ffcv),
                ]
            )

            ordering = ffcv.loader.OrderOption.RANDOM if self.is_train else ffcv.loader.OrderOption.SEQUENTIAL

            return ffcv.Loader(
                dataset_filepath,
                batch_size=batch_size,
                num_workers=dataloader_hparams.num_workers,
                order=ordering,
                distributed=False,
                pipelines={"image": image_pipeline, "label": label_pipeline},
                batches_ahead=dataloader_hparams.prefetch_factor,
                drop_last=self.drop_last,
            )
        else:
            if self.datadir is None:
                raise ValueError("datadir is required if use_synthetic is False")

            cifar10_mean = 0.4914, 0.4822, 0.4465
            cifar10_std = 0.247, 0.243, 0.261

            if self.is_train:
                transformation = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(cifar10_mean, cifar10_std),
                    ]
                )

                def label_noise(z):
                    def sub_func(x):
                        if torch.rand(1) < z:
                            # label=torch.zeros(x.size(-1))
                            # label[torch.randint(0,10,(1,))]=1
                            while True:
                                new_label = torch.randint(0, 10, (1,)).ravel()
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
                        lambda x: torch.nn.functional.one_hot(x, 10).ravel(),
                    ]
                )
            else:
                transformation = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(cifar10_mean, cifar10_std),
                    ]
                )

            dataset = CIFAR10(
                self.datadir,
                train=self.is_train,
                download=self.download,
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
class CIFAR10SubsetDatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the CIFAR-10 dataset for image classification from a local disk.

    Args:
        download (bool): Whether to download the dataset, if needed. Default: ``True``.
        use_ffcv (bool): Whether to use FFCV dataloaders. Default: ``False``.
        ffcv_dir (str): A directory containing train/val <file>.ffcv files. If these files don't exist and
            ``ffcv_write_dataset`` is ``True``, train/val <file>.ffcv files will be created in this dir. Default: ``"/tmp"``.
        ffcv_dest (str): <file>.ffcv file that has dataset samples. Default: ``"cifar_train.ffcv"``.
        ffcv_write_dataset (std): Whether to create dataset in FFCV format (<file>.ffcv) if it doesn't exist. Default:
        ``False``.
    """

    download: bool = hp.optional("whether to download the dataset, if needed", default=True)
    use_ffcv: bool = hp.optional("whether to use ffcv for faster dataloading", default=False)
    ffcv_dir: str = hp.optional(
        "A directory containing train/val <file>.ffcv files. If these files don't exist and ffcv_write_dataset is true, train/val <file>.ffcv files will be created in this dir.",
        default="/tmp",
    )
    ffcv_dest: str = hp.optional("<file>.ffcv file that has dataset samples", default="cifar_train.ffcv")
    ffcv_write_dataset: bool = hp.optional(
        "Whether to create dataset in FFCV format (<file>.ffcv) if it doesn't exist", default=False
    )
    size: int = hp.optional("", default=-1)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams, indices: List[int] = None):
        if self.size > 0:
            np.random.seed(0)
            indices = np.arange(50000, dtype="uint64")
            np.random.shuffle(indices)
            indices = indices[: self.size]
        if self.use_synthetic:
            total_dataset_size = 50_000 if self.is_train else 10_000
            dataset = SyntheticBatchPairDataset(
                total_dataset_size=total_dataset_size,
                data_shape=[3, 32, 32],
                num_classes=10,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )

        elif self.use_ffcv:
            try:
                import ffcv  # type: ignore
                from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder  # type: ignore
                from ffcv.pipeline.operation import Operation  # type: ignore
            except ImportError:
                raise ImportError(
                    textwrap.dedent(
                        """\
                    Composer was installed without ffcv support.
                    To use ffcv with Composer, please install ffcv in your environment."""
                    )
                )

            dataset_filepath = os.path.join(self.ffcv_dir, self.ffcv_dest)
            # always create if ffcv_write_dataset is true
            if self.ffcv_write_dataset:
                if dist.get_local_rank() == 0:
                    if self.datadir is None:
                        raise ValueError(
                            "datadir is required if use_synthetic is False and ffcv_write_dataset is True."
                        )
                    ds = CIFAR10(
                        self.datadir,
                        train=self.is_train,
                        download=self.download,
                    )

                    write_ffcv_dataset(dataset=ds, write_path=dataset_filepath)

                # Wait for the local rank 0 to be done creating the dataset in ffcv format.
                dist.barrier()

            if not os.path.exists(dataset_filepath):
                raise ValueError(
                    f"Dataset file containing samples not found at {dataset_filepath}. Use ffcv_dir flag to point to a dir containing {dataset_filepath}."
                )

            # Please note that this mean/std is different from the mean/std used for regular PyTorch dataloader as
            # ToTensor does the normalization for PyTorch dataloaders.
            cifar10_mean_ffcv = [125.307, 122.961, 113.8575]
            cifar10_std_ffcv = [51.5865, 50.847, 51.255]
            label_pipeline: List[Operation] = [IntDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.Squeeze()]
            image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

            if self.is_train:
                image_pipeline.extend(
                    [
                        ffcv.transforms.RandomHorizontalFlip(),
                        ffcv.transforms.RandomTranslate(padding=2, fill=tuple(map(int, cifar10_mean_ffcv))),
                        ffcv.transforms.Cutout(4, tuple(map(int, cifar10_mean_ffcv))),
                    ]
                )
            # Common transforms for train and test
            image_pipeline.extend(
                [
                    ffcv.transforms.ToTensor(),
                    ffcv.transforms.ToTorchImage(channels_last=False, convert_back_int16=False),
                    ffcv.transforms.Convert(torch.float32),
                    transforms.Normalize(cifar10_mean_ffcv, cifar10_std_ffcv),
                ]
            )

            ordering = ffcv.loader.OrderOption.RANDOM if self.is_train else ffcv.loader.OrderOption.SEQUENTIAL

            return ffcv.Loader(
                dataset_filepath,
                batch_size=batch_size,
                num_workers=dataloader_hparams.num_workers,
                order=ordering,
                distributed=False,
                pipelines={"image": image_pipeline, "label": label_pipeline},
                batches_ahead=dataloader_hparams.prefetch_factor,
                drop_last=self.drop_last,
            )
        else:
            if self.datadir is None:
                raise ValueError("datadir is required if use_synthetic is False")

            cifar10_mean = 0.4914, 0.4822, 0.4465
            cifar10_std = 0.247, 0.243, 0.261

            if self.is_train:
                transformation = transforms.Compose(
                    [
                        # transforms.RandomCrop(32, padding=4),
                        # transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(cifar10_mean, cifar10_std),
                    ]
                )
            else:
                transformation = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(cifar10_mean, cifar10_std),
                    ]
                )

            dataset = CIFAR10(
                self.datadir,
                train=self.is_train,
                download=self.download,
                transform=transformation,
            )

        if indices is not None:
            dataset = torch.utils.data.Subset(dataset, indices)

        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)

        return dataloader_hparams.initialize_object(
            dataset, batch_size=batch_size, sampler=sampler, drop_last=self.drop_last
        )


from itertools import repeat


def repeater(data_loader, times=200):
    return repeat(data_loader, times=times)


@dataclass
class CIFAR10SubsetTrainDatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the CIFAR-10 dataset for image classification from a local disk.

    Args:
        download (bool): Whether to download the dataset, if needed. Default: ``True``.
        use_ffcv (bool): Whether to use FFCV dataloaders. Default: ``False``.
        ffcv_dir (str): A directory containing train/val <file>.ffcv files. If these files don't exist and
            ``ffcv_write_dataset`` is ``True``, train/val <file>.ffcv files will be created in this dir. Default: ``"/tmp"``.
        ffcv_dest (str): <file>.ffcv file that has dataset samples. Default: ``"cifar_train.ffcv"``.
        ffcv_write_dataset (std): Whether to create dataset in FFCV format (<file>.ffcv) if it doesn't exist. Default:
        ``False``.
    """

    download: bool = hp.optional("whether to download the dataset, if needed", default=True)
    use_ffcv: bool = hp.optional("whether to use ffcv for faster dataloading", default=False)
    ffcv_dir: str = hp.optional(
        "A directory containing train/val <file>.ffcv files. If these files don't exist and ffcv_write_dataset is true, train/val <file>.ffcv files will be created in this dir.",
        default="/tmp",
    )
    ffcv_dest: str = hp.optional("<file>.ffcv file that has dataset samples", default="cifar_train.ffcv")
    ffcv_write_dataset: bool = hp.optional(
        "Whether to create dataset in FFCV format (<file>.ffcv) if it doesn't exist", default=False
    )
    size: int = hp.optional("", default=-1)
    times: int = hp.optional("", default=200)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams, indices: List[int] = None):
        if self.size > 0:
            np.random.seed(0)
            indices = np.arange(50000, dtype="uint64")
            np.random.shuffle(indices)
            indices = indices[: self.size]
        if self.use_synthetic:
            total_dataset_size = 50_000 if self.is_train else 10_000
            dataset = SyntheticBatchPairDataset(
                total_dataset_size=total_dataset_size,
                data_shape=[3, 32, 32],
                num_classes=10,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )

        elif self.use_ffcv:
            try:
                import ffcv  # type: ignore
                from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder  # type: ignore
                from ffcv.pipeline.operation import Operation  # type: ignore
            except ImportError:
                raise ImportError(
                    textwrap.dedent(
                        """\
                    Composer was installed without ffcv support.
                    To use ffcv with Composer, please install ffcv in your environment."""
                    )
                )

            dataset_filepath = os.path.join(self.ffcv_dir, self.ffcv_dest)
            # always create if ffcv_write_dataset is true
            if self.ffcv_write_dataset:
                if dist.get_local_rank() == 0:
                    if self.datadir is None:
                        raise ValueError(
                            "datadir is required if use_synthetic is False and ffcv_write_dataset is True."
                        )
                    ds = CIFAR10(
                        self.datadir,
                        train=self.is_train,
                        download=self.download,
                    )

                    write_ffcv_dataset(dataset=ds, write_path=dataset_filepath)

                # Wait for the local rank 0 to be done creating the dataset in ffcv format.
                dist.barrier()

            if not os.path.exists(dataset_filepath):
                raise ValueError(
                    f"Dataset file containing samples not found at {dataset_filepath}. Use ffcv_dir flag to point to a dir containing {dataset_filepath}."
                )

            # Please note that this mean/std is different from the mean/std used for regular PyTorch dataloader as
            # ToTensor does the normalization for PyTorch dataloaders.
            cifar10_mean_ffcv = [125.307, 122.961, 113.8575]
            cifar10_std_ffcv = [51.5865, 50.847, 51.255]
            label_pipeline: List[Operation] = [IntDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.Squeeze()]
            image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

            if self.is_train:
                image_pipeline.extend(
                    [
                        ffcv.transforms.RandomHorizontalFlip(),
                        ffcv.transforms.RandomTranslate(padding=2, fill=tuple(map(int, cifar10_mean_ffcv))),
                        ffcv.transforms.Cutout(4, tuple(map(int, cifar10_mean_ffcv))),
                    ]
                )
            # Common transforms for train and test
            image_pipeline.extend(
                [
                    ffcv.transforms.ToTensor(),
                    ffcv.transforms.ToTorchImage(channels_last=False, convert_back_int16=False),
                    ffcv.transforms.Convert(torch.float32),
                    transforms.Normalize(cifar10_mean_ffcv, cifar10_std_ffcv),
                ]
            )

            ordering = ffcv.loader.OrderOption.RANDOM if self.is_train else ffcv.loader.OrderOption.SEQUENTIAL

            return ffcv.Loader(
                dataset_filepath,
                batch_size=batch_size,
                num_workers=dataloader_hparams.num_workers,
                order=ordering,
                distributed=False,
                pipelines={"image": image_pipeline, "label": label_pipeline},
                batches_ahead=dataloader_hparams.prefetch_factor,
                drop_last=self.drop_last,
            )
        else:
            if self.datadir is None:
                raise ValueError("datadir is required if use_synthetic is False")

            cifar10_mean = 0.4914, 0.4822, 0.4465
            cifar10_std = 0.247, 0.243, 0.261

            if self.is_train:
                transformation = transforms.Compose(
                    [
                        # transforms.RandomCrop(32, padding=4),
                        # transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(cifar10_mean, cifar10_std),
                    ]
                )
            else:
                transformation = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(cifar10_mean, cifar10_std),
                    ]
                )

            dataset = CIFAR10(
                self.datadir,
                train=self.is_train,
                download=self.download,
                transform=transformation,
            )

        if indices is not None:
            dataset = torch.utils.data.Subset(dataset, indices)

        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)

        return list(
            repeater(
                list(
                    dataloader_hparams.initialize_object(
                        dataset, batch_size=batch_size, sampler=sampler, drop_last=self.drop_last
                    )
                )[0],
                times=self.times,
            )
        )


import numpy as np
from torchvision.datasets import CIFAR10
from typing import Tuple
from torch.utils.data.dataset import TensorDataset
import os
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision

os.environ["DATASETS"] = "/mnt/fs6/fengc/data"
DATASETS_FOLDER = "/mnt/fs6/fengc/data"


def center(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(0)
    return X_train - mean, X_test - mean


def standardize(X_train: np.ndarray, X_test: np.ndarray):
    std = X_train.std(0)
    return (X_train / std, X_test / std)


def flatten(arr: np.ndarray):
    return arr.reshape(arr.shape[0], -1)


def unflatten(arr: np.ndarray, shape: Tuple):
    return arr.reshape(arr.shape[0], *shape)


def _one_hot(tensor: Tensor, num_classes: int, default=0):
    M = F.one_hot(tensor, num_classes)
    M[M == 0] = default
    return M.float()


def make_labels(y, loss):
    if loss == "ce":
        return y
    elif loss == "mse":
        return _one_hot(y, 10, 0)


def load_cifar(loss: str) -> (TensorDataset, TensorDataset):
    cifar10_train = CIFAR10(root=DATASETS_FOLDER, download=True, train=True)
    cifar10_test = CIFAR10(root=DATASETS_FOLDER, download=True, train=False)
    X_train, X_test = flatten(cifar10_train.data / 255), flatten(cifar10_test.data / 255)
    y_train, y_test = make_labels(torch.tensor(cifar10_train.targets), loss), make_labels(
        torch.tensor(cifar10_test.targets), loss
    )
    center_X_train, center_X_test = center(X_train, X_test)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    train = TensorDataset(
        torch.from_numpy(unflatten(standardized_X_train, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_train
    )
    test = TensorDataset(
        torch.from_numpy(unflatten(standardized_X_test, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_test
    )
    return train, test


def load_cifar_downsampled(loss: str, size) -> (TensorDataset, TensorDataset):
    cifar10_train = CIFAR10(
        root=DATASETS_FOLDER, download=True, train=True, transform=torchvision.transforms.Resize(size)
    )
    cifar10_test = CIFAR10(
        root=DATASETS_FOLDER, download=True, train=False, transform=torchvision.transforms.Resize(size)
    )

    X_train, X_test = flatten(cifar10_train.data / 255), flatten(cifar10_test.data / 255)
    y_train, y_test = make_labels(torch.tensor(cifar10_train.targets), loss), make_labels(
        torch.tensor(cifar10_test.targets), loss
    )
    center_X_train, center_X_test = center(X_train, X_test)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    train = TensorDataset(
        torch.from_numpy(unflatten(standardized_X_train, (size, size, 3)).transpose((0, 3, 1, 2))).float(), y_train
    )
    test = TensorDataset(
        torch.from_numpy(unflatten(standardized_X_test, (size, size, 3)).transpose((0, 3, 1, 2))).float(), y_test
    )
    return train, test


import torch
import numpy as np
from typing import Tuple
from torch.utils.data import TensorDataset
import torchvision
import torch.nn.functional as F

# from wikitext import load_wikitext_2

DATASETS = [
    "cifar10",
    "cifar10-1k",
    "cifar10-100-downsampled",
    "cifar10-1k-downsampled",
    "cifar10-2k",
    "cifar10-5k",
    "cifar10-10k",
    "cifar10-20k",
    "chebyshev-3-20",
    "chebyshev-4-20",
    "chebyshev-5-20",
    "linear-50-50",
    "random-5k",
    "random-20k",
    "cifar10-5k-ac",
    "cifar10-20k-ac",
]


def flatten(arr: np.ndarray):
    return arr.reshape(arr.shape[0], -1)


def unflatten(arr: np.ndarray, shape: Tuple):
    return arr.reshape(arr.shape[0], *shape)


def num_input_channels(dataset_name: str) -> int:
    if dataset_name.startswith("cifar10") or dataset_name.startswith("random"):
        return 3
    elif dataset_name == "fashion":
        return 1


def image_size(dataset_name: str) -> int:
    if "downsampled" in dataset_name:
        return 5
    if dataset_name.startswith("cifar10") or dataset_name.startswith("random"):
        return 32
    elif dataset_name == "fashion":
        return 28


def num_classes(dataset_name: str) -> int:
    if dataset_name.startswith("cifar10") or dataset_name.startswith("random"):
        return 10
    elif dataset_name == "fashion":
        return 10


def get_pooling(pooling: str):
    if pooling == "max":
        return torch.nn.MaxPool2d((2, 2))
    elif pooling == "average":
        return torch.nn.AvgPool2d((2, 2))
    else:
        raise NotImplementedError("unknown pooling: {}".format(pooling))


def num_pixels(dataset_name: str) -> int:
    return num_input_channels(dataset_name) * image_size(dataset_name) ** 2


def take_first(dataset: TensorDataset, num_to_keep: int):
    return TensorDataset(dataset.tensors[0][0:num_to_keep], dataset.tensors[1][0:num_to_keep])


def load_dataset(dataset_name: str, loss: str) -> (TensorDataset, TensorDataset):
    if dataset_name == "cifar10":
        return load_cifar(loss)
    elif dataset_name == "cifar10-1k":
        train, test = load_cifar(loss)
        return take_first(train, 1000), test
    elif dataset_name == "cifar10-1k-downsampled":
        train, test = load_cifar(loss)
        x, y = train.tensors
        x = F.interpolate(x, size=(5, 5), mode="bilinear")
        train = TensorDataset(x, y)
        x, y = test.tensors
        x = F.interpolate(x, size=(5, 5), mode="bilinear")
        test = TensorDataset(x, y)
        return take_first(train, 1000), test
    elif dataset_name == "cifar10-100-downsampled":
        train, test = load_cifar(loss)
        x, y = train.tensors
        x = F.interpolate(x, size=(5, 5), mode="bilinear")
        train = TensorDataset(x, y)
        x, y = test.tensors
        x = F.interpolate(x, size=(5, 5), mode="bilinear")
        test = TensorDataset(x, y)
        return take_first(train, 100), test
    elif dataset_name == "cifar10-2k":
        train, test = load_cifar(loss)
        return take_first(train, 2000), test
    elif dataset_name == "cifar10-5k":
        train, test = load_cifar(loss)
        return take_first(train, 5000), test
    elif dataset_name == "cifar10-10k":
        train, test = load_cifar(loss)
        return take_first(train, 10000), test
    elif dataset_name == "cifar10-20k":
        train, test = load_cifar(loss)
        return take_first(train, 20000), test
    elif dataset_name == "random-5k":
        train, test = load_cifar(loss)
        train = take_first(train, 5000)
        x, y = train.tensors
        x = torch.randn(x.size())
        train = TensorDataset(x, y)
        return train, test
    elif dataset_name == "random-20k":
        train, test = load_cifar(loss)
        train = take_first(train, 20000)
        x, y = train.tensors
        x = torch.randn(x.size())
        train = TensorDataset(x, y)
        return train, test
    elif dataset_name == "cifar10-5k-ac":
        train, test = load_cifar(loss)
        train = take_first(train, 5000)
        x, y = train.tensors
        x = torch.randn(x.size())
        train = TensorDataset(x, x.reshape(x.size(0), -1))
        x, y = test.tensors
        x = torch.randn(x.size())
        test = TensorDataset(x, x.reshape(x.size(0), -1))
        return train, test
    elif dataset_name == "cifar10-20k-ac":
        train, test = load_cifar(loss)
        train = take_first(train, 20000)
        x, y = train.tensors
        x = torch.randn(x.size())
        train = TensorDataset(x, x.reshape(x.size(0), -1))
        x, y = test.tensors
        x = torch.randn(x.size())
        test = TensorDataset(x, x.reshape(x.size(0), -1))
        return train, test


@dataclass
class CIFAR10Subset2DatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the CIFAR-10 dataset for image classification from a local disk.

    Args:
        download (bool): Whether to download the dataset, if needed. Default: ``True``.
        use_ffcv (bool): Whether to use FFCV dataloaders. Default: ``False``.
        ffcv_dir (str): A directory containing train/val <file>.ffcv files. If these files don't exist and
            ``ffcv_write_dataset`` is ``True``, train/val <file>.ffcv files will be created in this dir. Default: ``"/tmp"``.
        ffcv_dest (str): <file>.ffcv file that has dataset samples. Default: ``"cifar_train.ffcv"``.
        ffcv_write_dataset (std): Whether to create dataset in FFCV format (<file>.ffcv) if it doesn't exist. Default:
        ``False``.
    """

    download: bool = hp.optional("whether to download the dataset, if needed", default=True)
    size: int = hp.optional("", default=-1)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams, indices: List[int] = None):
        train, test = load_cifar("mse")
        train = take_first(train, self.size)
        if self.is_train:
            return torch.utils.data.DataLoader(train, batch_size=batch_size)
        else:
            return torch.utils.data.DataLoader(test, batch_size=batch_size)


@dataclass
class CIFARWebDatasetHparams(WebDatasetHparams):
    """Common functionality for CIFAR WebDatasets.

    Args:
        remote (str): S3 bucket or root directory where dataset is stored.
        name (str): Key used to determine where dataset is cached on local filesystem.
        n_train_samples (int): Number of training samples.
        n_val_samples (int): Number of validation samples.
        height (int): Sample image height in pixels. Default: ``32``.
        width (int): Sample image width in pixels. Default: ``32``.
        n_classes (int): Number of output classes.
        channel_means (list of float): Channel means for normalization.
        channel_stds (list of float): Channel stds for normalization.
    """

    remote: str = hp.optional("WebDataset S3 bucket name", default="")
    name: str = hp.optional("WebDataset local cache name", default="")

    n_train_samples: int = hp.optional("Number of samples in training split", default=0)
    n_val_samples: int = hp.optional("Number of samples in validation split", default=0)
    height: int = hp.optional("Image height", default=32)
    width: int = hp.optional("Image width", default=32)
    n_classes: int = hp.optional("Number of output classes", default=0)
    channel_means: List[float] = hp.optional("Mean per image channel", default=(0, 0, 0))
    channel_stds: List[float] = hp.optional("Std per image channel", default=(0, 0, 0))

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams):
        from composer.datasets.webdataset_utils import load_webdataset

        if self.is_train:
            split = "train"
            transform = transforms.Compose(
                [
                    transforms.RandomCrop((self.height, self.width), (self.height // 8, self.width // 8)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.channel_means, self.channel_stds),
                ]
            )
        else:
            split = "val"
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(self.channel_means, self.channel_stds),
                ]
            )
        preprocess = lambda dataset: dataset.decode("pil").map_dict(jpg=transform).to_tuple("jpg", "cls")
        dataset = load_webdataset(
            self.remote,
            self.name,
            split,
            self.webdataset_cache_dir,
            self.webdataset_cache_verbose,
            self.shuffle,
            self.shuffle_buffer,
            preprocess,
            dist.get_world_size(),
            dataloader_hparams.num_workers,
            batch_size,
            self.drop_last,
        )
        return dataloader_hparams.initialize_object(
            dataset, batch_size=batch_size, sampler=None, drop_last=self.drop_last
        )


@dataclass
class CIFAR10WebDatasetHparams(CIFARWebDatasetHparams):
    """Defines an instance of the CIFAR-10 WebDataset for image classification.

    Args:
        remote (str): S3 bucket or root directory where dataset is stored.
            Default: ``'s3://mosaicml-internal-dataset-cifar10'``.
        name (str): Key used to determine where dataset is cached on local filesystem. Default: ``'cifar10'``.
        n_train_samples (int): Number of training samples. Default: ``50000``.
        n_val_samples (int): Number of validation samples. Default: ``10000``.
        n_classes (int): Number of output classes. Default: ``10``.
        channel_means (list of float): Channel means for normalization. Default: ``(0.4914, 0.4822, 0.4465)``.
        channel_stds (list of float): Channel stds for normalization. Default: ``(0.247, 0.243, 0.261)``.
    """

    remote: str = hp.optional("WebDataset S3 bucket name", default="s3://mosaicml-internal-dataset-cifar10")
    name: str = hp.optional("WebDataset local cache name", default="cifar10")

    n_train_samples: int = hp.optional("Number of samples in training split", default=50_000)
    n_val_samples: int = hp.optional("Number of samples in validation split", default=10_000)
    n_classes: int = hp.optional("Number of output classes", default=10)
    channel_means: List[float] = hp.optional("Mean per image channel", default=(0.4914, 0.4822, 0.4465))
    channel_stds: List[float] = hp.optional("Std per image channel", default=(0.247, 0.243, 0.261))


@dataclass
class CIFAR20WebDatasetHparams(CIFARWebDatasetHparams):
    """Defines an instance of the CIFAR-20 WebDataset for image classification.

    Args:
        remote (str): S3 bucket or root directory where dataset is stored.
            Default: ``'s3://mosaicml-internal-dataset-cifar20'``.
        name (str): Key used to determine where dataset is cached on local filesystem. Default: ``'cifar20'``.
        n_train_samples (int): Number of training samples. Default: ``50000``.
        n_val_samples (int): Number of validation samples. Default: ``10000``.
        n_classes (int): Number of output classes. Default: ``20``.
        channel_means (list of float): Channel means for normalization. Default: ``(0.5071, 0.4867, 0.4408)``.
        channel_stds (list of float): Channel stds for normalization. Default: ``(0.2675, 0.2565, 0.2761)``.
    """

    remote: str = hp.optional("WebDataset S3 bucket name", default="s3://mosaicml-internal-dataset-cifar20")
    name: str = hp.optional("WebDataset local cache name", default="cifar20")

    n_train_samples: int = hp.optional("Number of samples in training split", default=50_000)
    n_val_samples: int = hp.optional("Number of samples in validation split", default=10_000)
    n_classes: int = hp.optional("Number of output classes", default=20)
    channel_means: List[float] = hp.optional("Mean per image channel", default=(0.5071, 0.4867, 0.4408))
    channel_stds: List[float] = hp.optional("Std per image channel", default=(0.2675, 0.2565, 0.2761))


@dataclass
class CIFAR100WebDatasetHparams(CIFARWebDatasetHparams):
    """Defines an instance of the CIFAR-100 WebDataset for image classification.

    Args:
        remote (str): S3 bucket or root directory where dataset is stored.
            Default: ``'s3://mosaicml-internal-dataset-cifar100'``.
        name (str): Key used to determine where dataset is cached on local filesystem. Default: ``'cifar100'``.
        n_train_samples (int): Number of training samples. Default: ``50000``.
        n_val_samples (int): Number of validation samples. Default: ``10000``.
        n_classes (int): Number of output classes. Default: ``100``.
        channel_means (list of float): Channel means for normalization. Default: ``(0.5071, 0.4867, 0.4408)``.
        channel_stds (list of float): Channel stds for normalization. Default: ``(0.2675, 0.2565, 0.2761)``.
    """

    remote: str = hp.optional("WebDataset S3 bucket name", default="s3://mosaicml-internal-dataset-cifar100")
    name: str = hp.optional("WebDataset local cache name", default="cifar100")

    n_train_samples: int = hp.optional("Number of samples in training split", default=50_000)
    n_val_samples: int = hp.optional("Number of samples in validation split", default=10_000)
    n_classes: int = hp.optional("Number of output classes", default=100)
    channel_means: List[float] = hp.optional("Mean per image channel", default=(0.5071, 0.4867, 0.4408))
    channel_stds: List[float] = hp.optional("Std per image channel", default=(0.2675, 0.2565, 0.2761))
