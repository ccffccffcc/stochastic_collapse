from typing import Dict, List

from composer.loggers import InMemoryLogger, InMemoryLoggerHparams, Logger, LoggerDestinationHparams, WandBLoggerHparams


def configure_wandb_logger_hparams(
    hparams: WandBLoggerHparams, group: str, config_dict: Dict, filter_keys: List[str], **kwargs
) -> WandBLoggerHparams:
    hparams.group = group
    hparams.config = {k: v for k, v in config_dict.items() if k not in filter_keys}
    for k, v in kwargs.items():
        hparams.config[k] = v
    return hparams


def summary_logger_hparams_exists(loggers: List[LoggerDestinationHparams]) -> bool:
    for logger_hparams in loggers:
        if isinstance(logger_hparams, InMemoryLoggerHparams):
            return True
    return False


def get_log(logger: Logger) -> Dict:
    for logger_dest in logger.destinations:
        if isinstance(logger_dest, InMemoryLogger):
            return logger_dest.data


def get_summary(logger: Logger) -> Dict:
    for logger_dest in logger.destinations:
        if isinstance(logger_dest, InMemoryLogger):
            return logger_dest.most_recent_values
