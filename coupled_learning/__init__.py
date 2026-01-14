"""Coupled learning framework for physical networks."""

__version__ = "0.1.0"

from .base import PhysicsBackend
from .backends.simple_backend import (
    SimpleMechanicalBackend,
    SimpleMembraneBackend,
    SimpleFlowBackend,
    create_3x3_membrane_grid,
    create_3x3_mechanical_grid
)
from .backends.pixelated_membrane_backend import PixelatedMembraneBackend
from .signals import (SignalType, SignalComputer,
                     create_patch_sampling, create_3x3_patch_sampling)
from .update_rules import UpdateRule, create_update_rule
from .engine import CoupledLearningEngine

__all__ = [
    'PhysicsBackend',
    'SimpleMechanicalBackend',
    'SimpleMembraneBackend',
    'SimpleFlowBackend',
    'PixelatedMembraneBackend',
    'create_3x3_membrane_grid',
    'create_3x3_mechanical_grid',
    'SignalType',
    'SignalComputer',
    'create_patch_sampling',
    'create_3x3_patch_sampling',
    'UpdateRule',
    'create_update_rule',
    'CoupledLearningEngine',
]
