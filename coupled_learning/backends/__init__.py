"""
Backend implementations for coupled learning.

All backends implement the PhysicsBackend interface from coupled_learning.backends.
"""

from ..base import PhysicsBackend
from .simple_backend import (
    SimpleMechanicalBackend,
    SimpleMembraneBackend,
    SimpleFlowBackend,
    create_3x3_membrane_grid,
    create_3x3_mechanical_grid,
)
from .pixelated_membrane_backend import PixelatedMembraneBackend

__all__ = [
    'PhysicsBackend',
    'SimpleMechanicalBackend',
    'SimpleMembraneBackend',
    'SimpleFlowBackend',
    'PixelatedMembraneBackend',
    'create_3x3_membrane_grid',
    'create_3x3_mechanical_grid',
]
