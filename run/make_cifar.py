"""Download CIFAR10 and CIFAR100 to location defined by environment variable DATADIR"""

import os

import dotenv
from torchvision.datasets import CIFAR10, CIFAR100

dotenv.load_dotenv()

DATADIR = os.environ["DATADIR"]
CIFAR10(os.path.join(DATADIR, "cifar10"), download=True)
CIFAR100(os.path.join(DATADIR, "cifar100"), download=True)
