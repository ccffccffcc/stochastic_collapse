import datetime
import os
from dataclasses import dataclass
from typing import List

import yahp as hp
from composer.trainer.devices import DeviceGPU
from composer.utils import dist

from sgdcollapse.exps.exp import Experiment
from sgdcollapse.utils import utils


@dataclass
class Dev(Experiment):
    test_str: str = hp.required("Test string")
    test_bool: bool = hp.required("Test bool")
    test_int: int = hp.optional("Test optional int", default=1)

    @property
    def non_id_fields(self) -> List[str]:
        return ["test_int"]

    def exp_name(self) -> str:
        return self.hash

    def run_name(self) -> str:
        return os.path.join(self.hash, f"run_{self.test_int}")

    def location(self, file_name: str = "") -> str:
        return os.path.join(os.environ["EXP_DIR"], self.run_name(), file_name)

    def run(self) -> None:
        device = DeviceGPU()
        if dist.get_world_size() > 1:
            dist.initialize_dist(device.dist_backend, datetime.timedelta(seconds=300))

        if dist.get_global_rank() == 0:
            utils.make_new_local_dir(self.location())
        dist.barrier()

        self.global_rank = dist.get_global_rank()
        self.local_rank = dist.get_local_rank()
        self.node_rank = dist.get_node_rank()
        self.world_size = dist.get_world_size()
        self.local_world_size = dist.get_local_world_size()
        if dist.get_world_size() > 1:
            self.master_addr = os.environ["MASTER_ADDR"]
            self.master_port = os.environ["MASTER_PORT"]
        self.init = dist.is_initialized()

        print(vars(self))
        with open(self.location(f"rank_{self.global_rank}.yaml"), "x") as f:
            f.write(str(vars(self)))
        dist.barrier()
