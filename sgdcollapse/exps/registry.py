from typing import Type

from sgdcollapse import utils
from sgdcollapse.exps.dev import Dev
from sgdcollapse.exps.exp import Experiment
from sgdcollapse.exps.train import TrainExperiment
from sgdcollapse.exps.finetune import FinetuneExperiment


experiment_registry: utils.registry.Registry[str, Type[Experiment]] = {
    "dev": Dev,
    "train": TrainExperiment,
    "finetune": FinetuneExperiment,
}
