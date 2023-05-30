# TODO: Docstrings
import datetime
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional

import torch
import yahp as hp
from sgdcollapse.algorithm import AlgorithmHparams, get_algorithm_registry
from composer.callbacks import CallbackHparams
from composer.core import Precision
from composer.datasets import DataLoaderHparams, DatasetHparams
from composer.loggers import (
    LoggerDestination,
    LoggerDestinationHparams,
    LogLevel,
    ObjectStoreLoggerHparams,
    WandBLoggerHparams,
    logger_registry,
)
from composer.models import ModelHparams
from composer.optim import OptimizerHparams, SchedulerHparams
from composer.trainer import Trainer
from composer.trainer.devices import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from composer.utils import ObjectStoreHparams, dist, reproducibility

from sgdcollapse import utils
from sgdcollapse.callbacks import callback_registry
from sgdcollapse.data import data_registry
from sgdcollapse.exps.exp import Experiment
from sgdcollapse.models import model_registry
from sgdcollapse.optim import optimizer_registry, scheduler_registry

hparams_registry = {
    "model": model_registry,
    "train_data": data_registry,
    "eval_data": data_registry,
    "optimizer": optimizer_registry,
    "schedulers": scheduler_registry,
    "algorithms": get_algorithm_registry(),
    "callbacks": callback_registry,
    "device": {"cpu": CPUDeviceHparams, "gpu": GPUDeviceHparams},
    "loggers": logger_registry,
}


@dataclass
class TrainExperiment(Experiment):
    hparams_registry = hparams_registry
    # Model hparams (required)
    model: ModelHparams = hp.required("Model hparams.")
    # Training data hparams (required)
    train_data: DatasetHparams = hp.required("Training data hparams.")
    train_batch_size: int = hp.required("Total training batch size (across devices and grad_accum steps).")
    # Training hparams (required)
    max_duration: str = hp.required("Maximum training duration as a timestring.")
    optimizer: OptimizerHparams = hp.required("Optimizer hparams.")
    schedulers: List[SchedulerHparams] = hp.required("Sequence of lr scheduler hparams.")
    # Evaluation hparams (optional)
    eval_data: Optional[DatasetHparams] = hp.optional("Evaluation data hparams", default=None)
    eval_batch_size: Optional[int] = hp.optional("Total evaluation batch size, required if evaluating", default=None)
    eval_interval: Optional[str] = hp.optional("Timestring for evalualtion interval, None: every epoch", default=None)
    # Experiment hparams (optional)
    replicate: int = hp.optional("Experiment replicate number", default=0)
    seed: int = hp.optional("RNG seed for model init and SGD noise = (seed * (replicate + 1))", default=1)
    # Training hparams (optional)
    algorithms: List[AlgorithmHparams] = hp.optional("Algorithm hparams (Default: []).", default_factory=list)
    callbacks: List[CallbackHparams] = hp.optional("Callback hparams (Default: [])", default_factory=list)
    scale_schedule_ratio: Optional[float] = hp.optional("Scale training duration and lr schedule", default=None)
    grad_clip_norm: Optional[float] = hp.optional("Clip gradients to max norm", default=None)
    # Non-id hparams (optional)
    dataloader: DataLoaderHparams = hp.optional("Data loading hparams", default=DataLoaderHparams())
    device: Optional[DeviceHparams] = hp.optional("Training device, None: gpu if available, else cpu", default=None)
    precision: Precision = hp.optional("Numerical precision", default=Precision.AMP)
    save_interval: Optional[str] = hp.optional("Timestring for interval to save last state, None: 1ep", default=None)
    loggers: List[LoggerDestinationHparams] = hp.optional("Hparams for loggers (Default: []).", default_factory=list)
    object_store: Optional[ObjectStoreHparams] = hp.optional("Hparams for connecting to an object store", default=None)

    @property
    def non_id_fields(self) -> List[str]:
        return ["replicate", "dataloader", "device", "precision", "save_interval", "loggers", "object_store"]

    def exp_name(self) -> str:
        return self.hash

    def run_name(self) -> str:
        return os.path.join(self.hash, f"replicate_{self.replicate}")

    def location(self, file_name: str = "") -> str:
        return os.path.join("exps", self.run_name(), file_name)

    def validate(self) -> None:
        super().validate()
        world_size = dist.get_world_size()
        # Train batch size must be divisible by the number of processes
        if self.train_batch_size % world_size != 0:
            raise ValueError(f"Can't split train batch (size={self.train_batch_size}) into ({world_size}) processes")
        # Eval batch size must be specified if evaluating
        if self.eval_data is not None and self.eval_batch_size is None:
            raise ValueError("eval_batch_size must be specified if eval_data is specified")
        # If evaluating, eval batch size must be divisible by the number of processes
        if self.eval_batch_size is not None and self.eval_batch_size % world_size != 0:
            raise ValueError(f"Can't split eval batch (size={self.eval_batch_size}) into ({world_size}) processes")
        # Experiment replicate must be a non-negative integer
        if self.replicate < 0:
            raise ValueError(f"Replicate ({self.replicate}) must be non-negative")
        # Seed must be a positive integer
        if self.seed <= 0:
            raise ValueError(f"Seed ({self.seed}) must be positive")
        # If provided, scale schedule ratio must be positive
        if self.scale_schedule_ratio is not None and self.scale_schedule_ratio <= 0:
            raise ValueError(f"Scale schedule ratio ({self.scale_schedule_ratio}) must be positive")
        # Loggers must contain hparams for summary logger
        if not utils.logger.summary_logger_hparams_exists(self.loggers):
            raise ValueError("Loggers must contain hparams for summary logger")

    def _configure_loggers(self) -> List[LoggerDestination]:
        loggers = []
        for logger_hparams in self.loggers:
            if isinstance(logger_hparams, WandBLoggerHparams):
                logger_hparams = utils.logger.configure_wandb_logger_hparams(
                    deepcopy(logger_hparams),
                    self.exp_name(),
                    self.to_dict(),
                    self.non_id_fields,
                    replicate=self.replicate,
                )
            loggers.append(logger_hparams.initialize_object())
        if self.object_store is not None:
            loggers.append(ObjectStoreLoggerHparams(self.object_store).initialize_object())
        return loggers

    def run(self) -> None:
        self.validate()

        # If run completed and saved in object store, return from all ranks
        object_store = self.object_store.initialize_object() if self.object_store is not None else None
        if utils.object_exists(self.location("summary.pt"), object_store):
            print(f"Experiment completed and saved at {self.location()}")
            # return

        # Device
        device_hparams = self.device
        if device_hparams is None:
            device_hparams = GPUDeviceHparams() if torch.cuda.is_available() else CPUDeviceHparams()
        device = device_hparams.initialize_object()

        # TODO: What does this do?
        # Initialize distributed with default dist_timeout=300.0s
        if dist.get_world_size() > 1:
            dist.initialize_dist(device.dist_backend, datetime.timedelta(seconds=300.0))

        # Make new local run directory and save hparams locally from global rank 0
        if dist.get_global_rank() == 0:
            utils.make_new_local_dir(self.location())
            with open(self.location("hparams.yaml"), "w") as f:
                f.write(self.to_yaml())
        dist.barrier()

        # Reproducibility: Replicate seed = experiment seed * (replicate number + 1)
        # Seed rngs before initializing model for determinism, all ranks get the same seed
        seed = self.seed * (self.replicate + 1)
        reproducibility.seed_all(seed)

        # Initialize model and save locally from global rank 0
        model = self.model.initialize_object()
        if dist.get_global_rank() == 0:
            torch.save(model.module.state_dict(), self.location("model_init.pt"))
        dist.barrier()

        # Train data
        train_device_batch_size = self.train_batch_size // dist.get_world_size()
        train_dataloader = self.train_data.initialize_object(train_device_batch_size, self.dataloader)

        # Eval data
        eval_dataloader = None
        if self.eval_data is not None:
            eval_device_batch_size = self.eval_batch_size // dist.get_world_size()
            eval_dataloader = self.eval_data.initialize_object(eval_device_batch_size, self.dataloader)

        # Optimizer and schedulers
        optimizer = self.optimizer.initialize_object(model.parameters())
        schedulers = [scheduler.initialize_object() for scheduler in self.schedulers]

        # Algorithms and callbacks
        algorithms = [algorithm.initialize_object() for algorithm in self.algorithms]
        callbacks = [callback.initialize_object() for callback in self.callbacks]

        # Loggers
        loggers = self._configure_loggers()

        # Set defaults
        scale_schedule_ratio = self.scale_schedule_ratio if self.scale_schedule_ratio is not None else 1.0
        eval_interval = self.eval_interval if self.eval_interval is not None else 1
        grad_clip_norm = self.grad_clip_norm if self.grad_clip_norm is not None else -1.0
        save_interval = self.save_interval if self.save_interval is not None else "5000ep"

        # If state_last exists on the object store, resume training
        load_path, load_object_store = None, None
        # if utils.object_exists(self.location("state_last.pt"), object_store):
        #     load_path = self.location("state_last.pt")
        #     load_object_store = object_store

        # Trainer
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            max_duration=self.max_duration,
            algorithms=algorithms,
            optimizers=optimizer,
            schedulers=schedulers,
            scale_schedule_ratio=scale_schedule_ratio,
            step_schedulers_every_batch=True,
            eval_dataloader=eval_dataloader,
            eval_interval=eval_interval,
            callbacks=callbacks,
            loggers=loggers,
            run_name=self.run_name(),
            progress_bar=False,
            log_to_console=False,
            load_path=load_path,
            load_object_store=load_object_store,
            save_folder="exps/{run_name}",
            save_filename="state_last.pt",
            save_artifact_name="exps/{run_name}/state_last.pt",
            save_latest_filename=None,
            save_latest_artifact_name=None,
            save_overwrite=True,
            save_interval=save_interval,
            device=device,
            precision=self.precision,
            seed=seed,
            grad_clip_norm=grad_clip_norm,
        )

        # Trainer initialized => log locally saved hparams and model_init to object_store if provided
        if dist.get_global_rank() == 0:
            trainer.logger.file_artifact(
                LogLevel.FIT,
                self.location("hparams.yaml"),
                self.location("hparams.yaml"),
                overwrite=True,
            )
            trainer.logger.file_artifact(
                LogLevel.FIT,
                self.location("model_init.pt"),
                self.location("model_init.pt"),
                overwrite=True,
            )
        dist.barrier()

        # Fit
        trainer.fit()

        # Wrapup from global rank 0
        if dist.get_global_rank() == 0:
            # Final model
            torch.save(model.module.state_dict(), self.location("model_final.pt"))
            trainer.logger.file_artifact(
                LogLevel.FIT,
                self.location("model_final.pt"),
                self.location("model_final.pt"),
                overwrite=True,
            )
            # Log
            torch.save(utils.logger.get_log(trainer.logger), self.location("log.pt"))
            trainer.logger.file_artifact(
                LogLevel.FIT,
                self.location("log.pt"),
                self.location("log.pt"),
                overwrite=True,
            )
            # Summary
            torch.save(utils.logger.get_summary(trainer.logger), self.location("summary.pt"))
            trainer.logger.file_artifact(
                LogLevel.FIT,
                self.location("summary.pt"),
                self.location("summary.pt"),
                overwrite=True,
            )
        dist.barrier()

        # Close trainer
        trainer.close()

        # If using object store, clean up local directory
        if dist.get_global_rank() == 0 and object_store is not None:
            shutil.rmtree(self.location())
        dist.barrier()

        return
