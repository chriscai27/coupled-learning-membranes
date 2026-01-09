"""Coupled learning framework for physical networks."""

__version__ = "0.1.0"

from .backends import PhysicsBackend
from .simple_backend import SimpleMechanicalBackend, SimpleFlowBackend
from .signals import (SignalType, SignalComputer, 
                     create_patch_sampling, create_3x3_patch_sampling)
from .update_rules import UpdateRule, create_update_rule
from .engine import CoupledLearningEngine

__all__ = [
    'PhysicsBackend',
    'SimpleMechanicalBackend',
    'SimpleFlowBackend',
    'SignalType',
    'SignalComputer',
    'create_patch_sampling',  # New main function!
    'create_3x3_patch_sampling',
    'UpdateRule',
    'create_update_rule',
    'CoupledLearningEngine',
]