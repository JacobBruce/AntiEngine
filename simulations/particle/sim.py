"""
Particle-scale simulation for the anti-universe model.

Default scenario: a cluster of positive-mass particles and a cluster of
negative-mass particles are placed on opposite sides of the origin.

Expected behaviour:
  - Positive cluster contracts (mutual attraction)
  - Negative cluster expands  (mutual repulsion)
  - The two clusters repel each other (pos-neg interaction)

Energy conservation is used as the primary correctness check — the leapfrog
integrator should keep total energy fluctuations < 0.1% over hundreds of steps.
"""

from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp

from antiengine import compute_energy
from antiengine.integrator import make_leapfrog_step


@dataclass
class ParticleSimConfig:
	n_positive: int   = 50      # number of positive-mass particles
	n_negative: int   = 50      # number of negative-mass particles
	mass_pos:   float = 1.0     # mass of each positive particle  (must be > 0)
	mass_neg:   float = -1.0    # mass of each negative particle  (must be < 0)
	G:          float = 1.0     # gravitational constant
	softening:  float = 0.3     # Plummer softening length (~inter-particle spacing for N=50, spread=1)
	dt:         float = 0.002   # time step
	spread:     float = 1.0     # Gaussian spread of initial clusters
	separation: float = 5.0     # half the initial centre-to-centre distance
	vel_scale:  float = 0.1     # initial velocity dispersion
	seed:       int   = 42      # random seed


def initialize_particles(config: ParticleSimConfig):
	"""
	Create initial positions, velocities, and masses for the particle simulation.

	Layout:
	  - Positive particles clustered at (-separation, 0)
	  - Negative particles clustered at (+separation, 0)

	Returns:
		positions  : (N, 2) jax float32 array
		velocities : (N, 2) jax float32 array
		masses     : (N,)   jax float32 array  (signed)
	"""
	rng = np.random.default_rng(config.seed)
	N = config.n_positive + config.n_negative

	# Two Gaussian clusters centred on opposite sides of the origin
	pos_positive = rng.normal(
		loc=[-config.separation, 0.0], scale=config.spread,
		size=(config.n_positive, 2),
	)
	pos_negative = rng.normal(
		loc=[+config.separation, 0.0], scale=config.spread,
		size=(config.n_negative, 2),
	)
	positions = np.vstack([pos_positive, pos_negative])

	# Small random velocities for both groups
	velocities = rng.normal(scale=config.vel_scale, size=(N, 2))

	# Positive masses first, then negative
	masses = np.concatenate([
		np.full(config.n_positive, config.mass_pos),
		np.full(config.n_negative, config.mass_neg),
	])

	return (
		jnp.array(positions, dtype=jnp.float64),
		jnp.array(velocities, dtype=jnp.float64),
		jnp.array(masses,     dtype=jnp.float64),
	)


def run_simulation(
	config: ParticleSimConfig,
	n_steps:      int = 500,
	record_every: int = 5,
):
	"""
	Run the particle simulation and return a state history.

	Each simulation step runs on the GPU via a JIT-compiled leapfrog integrator.
	State is sampled every `record_every` steps and transferred to CPU for storage.

	Returns a dict with keys:
	  'positions'  : list of (N, 2) numpy arrays
	  'velocities' : list of (N, 2) numpy arrays
	  'KE'         : list of float
	  'PE'         : list of float
	  'total_E'    : list of float
	  'time'       : list of float
	  'masses'     : (N,) numpy array  (fixed throughout)
	  'config'     : the ParticleSimConfig used
	"""
	positions, velocities, masses = initialize_particles(config)
	step_fn = make_leapfrog_step(masses, config.G, config.softening, config.dt)

	history = {
		'positions':  [],
		'velocities': [],
		'KE':         [],
		'PE':         [],
		'total_E':    [],
		'time':       [],
		'masses':     np.array(masses),
		'config':     config,
	}

	print(f"Running {n_steps} steps  (recording every {record_every})")
	print(f"  N_pos={config.n_positive}  N_neg={config.n_negative}  "
	      f"G={config.G}  dt={config.dt}  softening={config.softening}")

	# Trigger JIT compilation before the timed loop
	_ = step_fn(positions, velocities)
	print("  JIT compilation complete — starting simulation...")

	t = 0.0
	for i in range(n_steps):
		positions, velocities = step_fn(positions, velocities)
		t += config.dt

		if i % record_every == 0:
			KE, PE, E_total = compute_energy(
				positions, velocities, masses, config.G, config.softening
			)
			history['positions'].append(np.array(positions))
			history['velocities'].append(np.array(velocities))
			history['KE'].append(float(KE))
			history['PE'].append(float(PE))
			history['total_E'].append(float(E_total))
			history['time'].append(t)

	print(f"  Done. Recorded {len(history['positions'])} frames.")
	return history


def print_energy_diagnostics(history):
	"""Print an energy conservation summary to stdout."""
	E = np.array(history['total_E'])
	E0 = E[0]
	drift      = (E[-1] - E0) / abs(E0) * 100 if E0 != 0 else float('nan')
	rms_fluct  = np.std(E)   / abs(E0) * 100 if E0 != 0 else float('nan')
	max_fluct  = np.max(np.abs(E - E0)) / abs(E0) * 100 if E0 != 0 else float('nan')

	print("\n--- Energy diagnostics ---")
	print(f"  Initial total E : {E0:.6f}")
	print(f"  Final total E   : {E[-1]:.6f}")
	print(f"  Relative drift  : {drift:.4f}%")
	print(f"  RMS fluctuation : {rms_fluct:.4f}%")
	print(f"  Max fluctuation : {max_fluct:.4f}%")
	print("--------------------------")
