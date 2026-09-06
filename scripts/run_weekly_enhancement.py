#!/usr/bin/env python3
"""
Weekly Enhancement Runner
Called by the weekly-enhance.yml workflow
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.enhancement.param_tuner import run_weekly_enhancement

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

result = run_weekly_enhancement()
print("Enhancement result:", result)
