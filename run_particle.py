"""
Entry point for the particle-scale anti-universe simulation.

Run from the project root:
    python run_particle.py

Adjust the config values below to explore different scenarios.
"""

from simulations.particle.sim import ParticleSimConfig, run_simulation, print_energy_diagnostics
from simulations.particle.visualize import animate_simulation


config = ParticleSimConfig(
	n_positive = 50,
	n_negative = 50,
	G          = 1.0,
	softening  = 0.3,
	dt         = 0.002,
	spread     = 1.0,
	separation = 5.0,
	vel_scale  = 0.1,
	seed       = 42,
)

history = run_simulation(config, n_steps=1000, record_every=5)
print_energy_diagnostics(history)
animate_simulation(history, interval=40)
