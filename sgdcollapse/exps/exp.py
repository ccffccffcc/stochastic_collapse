import abc
from dataclasses import dataclass
from typing import List

import yahp as hp

from sgdcollapse import utils


@dataclass
class Experiment(hp.Hparams, abc.ABC):
    @property
    def non_id_fields(self) -> List[str]:
        return []

    @property
    def description(self) -> str:
        return utils.hparams.description(self)

    @property
    def hash(self) -> str:
        return utils.hash(self.description)

    @abc.abstractmethod
    def exp_name(self) -> str:
        ...

    @abc.abstractmethod
    def run_name(self) -> str:
        ...

    @abc.abstractmethod
    def run(self) -> None:
        ...
