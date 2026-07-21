# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


os.environ.setdefault("MOIABC_SENSITIVITY_SUITES", "CEC2020_MMO")
os.environ.setdefault(
    "MOIABC_SENSITIVITY_FUNCTION_IDS",
    "MMF14,MMF15,MMF14_A,MMF15_A,MMF16_L1,MMF16_L2,MMF16_L3",
)
os.environ.setdefault("MOIABC_SENSITIVITY_OUTPUT_DIR", "moiabc_sensitivity_results_added_3obj")
os.environ.setdefault("MOIABC_SENSITIVITY_WORKERS", "8")

from multi_objective.run_moiabc_sensitivity import main  # noqa: E402


if __name__ == "__main__":
    main()
