"""Coupled learning framework for physical networks."""

__version__ = "0.1.0"

from .backends import PhysicsBackend
from .simple_backend import SimpleMechanicalBackend, SimpleFlowBackend

__all__ = [
    'PhysicsBackend',
    'SimpleMechanicalBackend', 
    'SimpleFlowBackend',
]