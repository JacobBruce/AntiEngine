"""
Entry point for the single-galaxy anti-universe simulation (Phase 2A).

Three interaction rule models for negative mass:
  1. CPT / Time-reversed: neg-neg repel, pos-neg mutual repulsion (default)
  2. Bimetric GR: neg-neg attract, pos-neg mutual repulsion
  3. Bondi / Standard Newton: neg-neg repel, pos-neg asymmetric (runaway)

Units: kpc, M☉, Gyr.

Run from the project root:
    python run_galaxy.py

Adjust the config below to explore different scenarios.
Key parameters:
  neg_mass_ratio  — ratio of total negative mass to total positive mass
                    (0 = no negative mass, 1 = 50% negative, 9 = 90% negative)
  n_negative      — number of negative-mass tracer particles
  n_steps / dt    — total simulation time = n_steps * dt  Gyr
"""

import copy
from simulations.galaxy.sim import run_galaxy_simulation, print_galaxy_diagnostics
from simulations.galaxy.initial_conditions import GalaxyConfig
from simulations.galaxy.visualize import animate_galaxy, plot_rotation_curve_comparison

base_config = GalaxyConfig(
	# Particle counts — increase for production runs (GPU recommended)
	n_disk     = 2000,
	n_bulge    = 500,
	n_halo     = 500,
	n_gas      = 500,
	n_negative = 3000,

	# Physics
    interaction_mode   = 'cpt',  # gravitational interaction rules
	neg_mass_ratio     = 1.0,    # |neg| / pos  (1.0 = 50/50)
	neg_bg_density     = 3.9e4,  # M☉/kpc³ — auto-derives neg_sphere_radius
	neg_density_slope  = 0.0,    # 0 = uniform density
    neg_vel_dispersion = 25.0,   # kpc/Gyr  (0 = start at rest)

	# Integration
	dt     = 0.002,  # Gyr (2 Myr per step)
	softening = 0.5, # kpc

	use_elastic_boundary      = False,
	use_reinjection_boundary  = True,
	hubble_expansion_boundary = True,
	cavity_from_boundary      = False,
)

N_STEPS = 500
RECORD_EVERY = 10
"""
MODES = ['cpt', 'bimetric', 'bondi']

histories = {}
for mode in MODES:
	print("=" * 70)
	print(f"  INTERACTION MODE: {mode.upper()}")
	print("=" * 70)
	config = copy.copy(base_config)
	config.interaction_mode = mode
	histories[mode] = run_galaxy_simulation(config, n_steps=N_STEPS, record_every=RECORD_EVERY)
	print_galaxy_diagnostics(histories[mode])
	print()

# Plot comparison for each mode
for mode in MODES:
	plot_rotation_curve_comparison(histories[mode])

# Animate the CPT run by default
animate_galaxy(histories['cpt'], interval=60)
"""

history = run_galaxy_simulation(base_config, n_steps=N_STEPS, record_every=RECORD_EVERY)
print_galaxy_diagnostics(history)
#plot_rotation_curve_comparison(history)
animate_galaxy(history, interval=60)
