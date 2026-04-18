"""
AntiEngine — physics simulations for the negative mass anti-universe model.
"""
import jax
# 64-bit floats throughout — essential for energy conservation in long integrations
jax.config.update("jax_enable_x64", True)

from .physics import compute_accelerations, compute_energy
from .integrator import make_leapfrog_step, make_elastic_boundary_fn

__all__ = ['compute_accelerations', 'compute_energy', 'make_leapfrog_step', 'make_elastic_boundary_fn']
