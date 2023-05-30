from typing import Type

import composer.optim as composer_optim

from sgdcollapse import utils

optimizer_registry: utils.registry.Registry[str, Type[composer_optim.OptimizerHparams]] = {
    "sgd": composer_optim.SGDHparams,
    "decoupled_sgdw": composer_optim.DecoupledSGDWHparams,
    "adam": composer_optim.AdamHparams,
}

scheduler_registry: utils.registry.Registry[str, Type[composer_optim.SchedulerHparams]] = {
    "constant": composer_optim.ConstantSchedulerHparams,
    "cosine_decay": composer_optim.CosineAnnealingSchedulerHparams,
    "cosine_decay_with_warmup": composer_optim.CosineAnnealingWithWarmupSchedulerHparams,
    "cosine_warmrestart": composer_optim.CosineAnnealingWarmRestartsSchedulerHparams,
    "linear_decay": composer_optim.LinearSchedulerHparams,
    "linear_decay_with_warmup": composer_optim.LinearWithWarmupSchedulerHparams,
    "multistep": composer_optim.MultiStepSchedulerHparams,
    "multistep_with_warmup": composer_optim.MultiStepWithWarmupSchedulerHparams,
}
