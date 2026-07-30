# -*- coding: utf-8 -*-

import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from multi_objective.run_multi_objective_comparison import (
    STANDARD_EXPERIMENT_GROUP,
    run_single_experiment_group,
)


if __name__ == "__main__":
    run_single_experiment_group(STANDARD_EXPERIMENT_GROUP)
