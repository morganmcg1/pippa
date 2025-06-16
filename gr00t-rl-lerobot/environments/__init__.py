"""
Environment wrappers for GR00T-RL experiments.

Provides different approaches for adapting Fetch to SO-101:
1. Cartesian-only (fetch_wrapper) - Quick start, 7-DoF hidden
2. Joint coupling (fetch_so101_coupled) - 6-DoF simulation
"""

from .fetch_wrapper import FetchToSO101Wrapper, make_fetch_so101_env
from .fetch_so101_coupled import FetchSO101CoupledWrapper, make_fetch_so101_coupled_env

__all__ = [
    "FetchToSO101Wrapper",
    "make_fetch_so101_env",
    "FetchSO101CoupledWrapper", 
    "make_fetch_so101_coupled_env",
]