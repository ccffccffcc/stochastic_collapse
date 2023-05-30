"""Entrypoint for running an Experiment

Usage: python run/exp.py EXP_TYPE [-f CONFIG_FILE] [--arg1=ARG1 --arg2=ARG2 ...] [--help] [--get_name]
"""

import sys
import warnings
from typing import Type

import dotenv
from composer.utils import dist

from sgdcollapse import utils
from sgdcollapse.exps import experiment_registry

dotenv.load_dotenv()


def warning_on_one_line(message: str, category: Type[Warning], filename: str, lineno: int, file=None, line=None):
    # From https://stackoverflow.com/questions/26430861/make-pythons-warnings-warn-not-mention-itself
    return f"{category.__name__}: {message} (source: {filename}:{lineno})\n"


def main() -> None:
    """Handle system arguments, create, and run an experiment.

    Must specify a valid experiment type as the first argument:
    $ python run/exp.py EXP_TYPE [-f CONFIG_FILE] [--arg1=ARG1 --arg2=ARG2 ...] [--help] [--get_name]

    If no arguments are provided or if the first argument isn't an experiment type from the experiment registry,
    run/exp.py will print the valid experiment types and exit.

    To print the cli help for a valid experiment type, provide just EXP_TYPE or use the --help flag:
    $ python run/exp.py EXP_TYPE
    $ python run/exp.py EXP_TYPE [-f CONFIG_FILE] [--arg1=ARG1 --arg2=ARG2 ...] --help

    If the --get_name flag is provided, run/exp.py will print the description
    of a successfully initialized experiment and exit without running it.

    Experiments do not need to implement EXP_TYPE and --get_name as fields, they are
    handled by run/exp.py and removed from sys.argv before the experiment is created.
    """
    warnings.formatwarning = warning_on_one_line

    # Handle system arguments
    experiment_types = utils.registry.options(experiment_registry)
    if len(sys.argv) == 1:
        sys.exit(f"\nExperiment type not specified, expected one of: {experiment_types}\n")
    exp_type = sys.argv.pop(1)
    if exp_type not in experiment_types:
        sys.exit(f"\nInvalid experiment type (1st argument): '{exp_type}', expected one of: {experiment_types}\n")
    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "--help"]
    get_name = False
    if "--get_name" in sys.argv:
        get_name = True
        sys.argv.pop(sys.argv.index("--get_name"))

    # Create experiment
    exp = experiment_registry[exp_type].create(cli_args=True)

    # Startup
    if dist.get_local_rank() == 0:  # Print experiment description on local rank 0 only
        print("\n" + ("=" * 80) + "\n\n" + exp.description + "\n\n" + exp.run_name() + "\n\n" + ("=" * 80) + "\n")
    if get_name:  # If get_name, exit from all ranks
        sys.exit(0)

    # Run experiment
    exp.run()

    # Wrapup
    if dist.get_local_rank() == 0:
        print("\n" + ("=" * 80) + "\n")


if __name__ == "__main__":
    main()
