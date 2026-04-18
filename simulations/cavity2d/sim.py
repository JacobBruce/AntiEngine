"""
2D Cavity Self-Limiting Verification Simulation

Places positive and negative mass particles in a randomized uniform distribution
inside a circular boundary. All particles reflect off the boundary walls. Cavities
form naturally around positive-mass concentrations.

Purpose: verify that the cavity self-limiting mechanism works — specifically,
test how cavity radius depends on the surrounding neg-mass density.
If self-limiting holds, the cavity should shrink with increasing ρ_neg so that
the effective mass (background deficit) remains ≈ M_gal. In 2D:
  M_eff = π ρ_bg R_cavity² − M_neg_enc(R_cavity) ≈ M_gal
implying R_cavity ∝ 1/√ρ_bg for a fully evacuated cavity.

Uses the existing AntiEngine direct-summation gravity (CPT interaction rules)
and leapfrog integrator.
"""

from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp

from antiengine.physics import compute_accelerations, compute_energy


@dataclass
class Cavity2DConfig:
	# Particle counts
	n_positive: int = 1			# number of positive-mass particles (galaxy seeds)
	n_negative: int = 500		# number of negative-mass particles

	# Masses (dimensionless units)
	mass_pos: float = 10.0		# mass of each positive particle
	mass_neg: float = -1.0		# mass of each negative particle (signed)

	# Domain
	boundary_radius: float = 20.0	# circular boundary radius

	# Background density — the true neg-mass density far from the galaxy.
	# If None, computed from n_negative / (π R²). If set explicitly, this is
	# the physical ρ_bg; the simulation boundary may be smaller than the true
	# cavity edge, so boundary pileup should be watched for.
	rho_bg: float = None

	# Physics
	G: float = 1.0
	softening: float = 0.5		# Plummer softening

	# Integration
	dt: float = 0.005
	n_steps: int = 10000
	record_every: int = 50

	# Initial conditions
	seed: int = 42
	vel_scale: float = 0.0		# initial velocity dispersion (0 = cold start)

	# Boundary
	restitution: float = 0.8	# velocity retained after wall bounce (1 = elastic)

	# Pin positive particles at origin (fixed galaxy)
	pin_positive: bool = True

	def get_rho_bg(self):
		"""Return the background density, either explicit or computed from N/area."""
		if self.rho_bg is not None:
			return self.rho_bg
		return self.n_negative / (np.pi * self.boundary_radius**2)


def initialize_particles(config: Cavity2DConfig):
	"""
	Create initial positions, velocities, and masses for the 2D cavity sim.

	All particles start in a uniform random distribution inside the circular
	boundary. Positive particles are placed near the center if pin_positive is set.

	Returns:
		positions  : (N, 2) jax float64 array
		velocities : (N, 2) jax float64 array
		masses     : (N,)   jax float64 array (signed)
	"""
	rng = np.random.default_rng(config.seed)
	N = config.n_positive + config.n_negative
	R = config.boundary_radius

	# Uniform distribution in a disk: sample r² ~ U(0, R²), θ ~ U(0, 2π)
	r_neg = np.sqrt(rng.uniform(0, R**2, config.n_negative))
	theta_neg = rng.uniform(0, 2 * np.pi, config.n_negative)
	pos_negative = np.column_stack([r_neg * np.cos(theta_neg), r_neg * np.sin(theta_neg)])

	# Positive particles: at origin if pinned, else random
	if config.pin_positive:
		if config.n_positive == 1:
			pos_positive = np.array([[0.0, 0.0]])
		else:
			# Small cluster near center
			r_pos = rng.normal(scale=0.5, size=config.n_positive)
			theta_pos = rng.uniform(0, 2 * np.pi, config.n_positive)
			pos_positive = np.column_stack([r_pos * np.cos(theta_pos), r_pos * np.sin(theta_pos)])
	else:
		r_pos = np.sqrt(rng.uniform(0, R**2, config.n_positive))
		theta_pos = rng.uniform(0, 2 * np.pi, config.n_positive)
		pos_positive = np.column_stack([r_pos * np.cos(theta_pos), r_pos * np.sin(theta_pos)])

	positions = np.vstack([pos_positive, pos_negative])

	# Initial velocities
	velocities = rng.normal(scale=max(config.vel_scale, 1e-10), size=(N, 2))
	if config.vel_scale == 0.0:
		velocities[:] = 0.0

	# Masses: positive first, then negative
	masses = np.concatenate([
		np.full(config.n_positive, config.mass_pos),
		np.full(config.n_negative, config.mass_neg),
	])

	return (
		jnp.array(positions, dtype=jnp.float64),
		jnp.array(velocities, dtype=jnp.float64),
		jnp.array(masses, dtype=jnp.float64),
	)


def _make_boundary_fn(masses, boundary_radius, restitution, pin_positive):
	"""
	Create a JIT-compiled circular boundary reflection function for 2D.

	All particles that exit the boundary are reflected. If pin_positive is set,
	positive particles are held at their initial position instead.
	"""
	pos_mask = jnp.array(masses > 0)

	@jax.jit
	def apply_boundary(positions, velocities):
		r = jnp.linalg.norm(positions, axis=1)  # (N,)
		r_hat = positions / jnp.maximum(r, 1e-10)[:, None]  # (N, 2)
		v_rad = jnp.sum(velocities * r_hat, axis=1)  # radial speed (N,)

		# Reflect particles outside boundary that are moving outward
		outside = (r > boundary_radius) & (v_rad > 0)
		new_vel = jnp.where(
			outside[:, None],
			velocities - (1.0 + restitution) * v_rad[:, None] * r_hat,
			velocities,
		)

		# Clamp positions back inside boundary
		r_clamped = jnp.minimum(r, boundary_radius)
		new_pos = jnp.where(
			(r > boundary_radius)[:, None],
			r_hat * r_clamped[:, None],
			positions,
		)

		# Pin positive particles at origin if requested
		if pin_positive:
			new_pos = jnp.where(pos_mask[:, None], jnp.zeros_like(new_pos), new_pos)
			new_vel = jnp.where(pos_mask[:, None], jnp.zeros_like(new_vel), new_vel)

		return new_pos, new_vel

	return apply_boundary


def _make_step_fn(masses, G, softening, dt):
	"""Create a JIT-compiled leapfrog step for 2D particles."""
	@jax.jit
	def step(positions, velocities):
		accels = compute_accelerations(positions, masses, G, softening, 'cpt')
		vel_half = velocities + 0.5 * dt * accels
		pos_new = positions + dt * vel_half
		accels_new = compute_accelerations(pos_new, masses, G, softening, 'cpt')
		vel_new = vel_half + 0.5 * dt * accels_new
		return pos_new, vel_new

	return step


def run_cavity_2d(config: Cavity2DConfig):
	"""
	Run the 2D cavity simulation and return state history.

	Returns dict with keys:
		'positions'  : list of (N, 2) numpy arrays
		'velocities' : list of (N, 2) numpy arrays
		'KE', 'PE', 'total_E' : energy time series
		'time'       : list of float
		'masses'     : (N,) numpy array
		'config'     : the Cavity2DConfig used
	"""
	positions, velocities, masses = initialize_particles(config)

	step_fn = _make_step_fn(masses, config.G, config.softening, config.dt)
	boundary_fn = _make_boundary_fn(
		np.array(masses), config.boundary_radius, config.restitution, config.pin_positive
	)

	history = {
		'positions': [],
		'velocities': [],
		'KE': [],
		'PE': [],
		'total_E': [],
		'time': [],
		'masses': np.array(masses),
		'config': config,
	}

	N = config.n_positive + config.n_negative
	print(f"=== 2D Cavity Self-Limiting Simulation ===")
	print(f"  N_pos={config.n_positive} (mass={config.mass_pos} each)")
	print(f"  N_neg={config.n_negative} (mass={config.mass_neg} each)")
	print(f"  Total pos mass: {config.n_positive * config.mass_pos:.1f}")
	print(f"  Total neg mass: {config.n_negative * abs(config.mass_neg):.1f}")
	print(f"  Boundary radius: {config.boundary_radius}")
	print(f"  G={config.G}  dt={config.dt}  softening={config.softening}")
	print(f"  Steps: {config.n_steps} (recording every {config.record_every})")

	# Warm up JIT
	_ = step_fn(positions, velocities)
	_ = boundary_fn(positions, velocities)
	print("  JIT compilation complete — starting simulation...")

	t = 0.0
	for i in range(config.n_steps):
		positions, velocities = step_fn(positions, velocities)
		positions, velocities = boundary_fn(positions, velocities)
		t += config.dt

		if i % config.record_every == 0:
			KE, PE, E_total = compute_energy(
				positions, velocities, masses, config.G, config.softening, 'cpt'
			)
			history['positions'].append(np.array(positions))
			history['velocities'].append(np.array(velocities))
			history['KE'].append(float(KE))
			history['PE'].append(float(PE))
			history['total_E'].append(float(E_total))
			history['time'].append(t)

			if i % (config.record_every * 20) == 0:
				# Measure cavity radius (distance to nearest neg particle from origin)
				neg_pos = np.array(positions[config.n_positive:])
				neg_r = np.linalg.norm(neg_pos, axis=1)
				cavity_r = np.percentile(neg_r, 5)  # 5th percentile as proxy
				print(f"  t={t:.2f} | cavity r~{cavity_r:.2f} | "
					  f"E={float(E_total):.2f}")

	# Final snapshot
	positions_final = np.array(positions)
	neg_pos = positions_final[config.n_positive:]
	neg_r = np.linalg.norm(neg_pos, axis=1)

	print(f"\n  Done. Recorded {len(history['positions'])} frames.")
	print(f"  Final cavity radius (5th pctl): {np.percentile(neg_r, 5):.2f}")
	print(f"  Final cavity radius (min r):    {neg_r.min():.2f}")

	return history


def measure_cavity_radius(history, method='percentile', percentile=5):
	"""
	Measure the cavity radius over time from the simulation history.

	Methods:
		'percentile' : radius enclosing the innermost X% of neg particles
		'density'    : radius where neg density drops below 50% of mean

	Returns:
		times : (n_frames,) array
		radii : (n_frames,) array of measured cavity radii
	"""
	n_pos = history['config'].n_positive
	times = np.array(history['time'])
	radii = np.zeros(len(times))

	for i, pos in enumerate(history['positions']):
		neg_pos = pos[n_pos:]
		neg_r = np.linalg.norm(neg_pos, axis=1)

		if method == 'percentile':
			radii[i] = np.percentile(neg_r, percentile)
		elif method == 'density':
			# Measure radial density profile and find where it drops
			r_bins = np.linspace(0, history['config'].boundary_radius, 50)
			r_centres = 0.5 * (r_bins[:-1] + r_bins[1:])
			areas = np.pi * (r_bins[1:]**2 - r_bins[:-1]**2)
			counts, _ = np.histogram(neg_r, bins=r_bins)
			density = counts / areas
			mean_density = history['config'].get_rho_bg()
			# Find first bin where density exceeds 50% of mean
			above = density > 0.5 * mean_density
			if np.any(above):
				radii[i] = r_centres[np.argmax(above)]
			else:
				radii[i] = 0.0

	return times, radii


def measure_density_profile(history, n_bins=80, equil_fraction=0.5):
	"""
	Extract the time-averaged radial density profile from equilibrium snapshots.

	Averages over the last `equil_fraction` of recorded frames to reduce noise.
	Returns surface number density (particles per unit area) vs radius.

	Parameters
	----------
	history : dict from run_cavity_2d
	n_bins : number of radial bins
	equil_fraction : fraction of frames (from the end) to average over

	Returns
	-------
	r_centres : (n_bins,) bin centre radii
	density   : (n_bins,) mean surface density (particles/area)
	density_std : (n_bins,) standard deviation across frames
	rho_bg    : background density (total particles / total area)
	"""
	config = history['config']
	n_pos = config.n_positive
	R = config.boundary_radius

	r_bins = np.linspace(0, R, n_bins + 1)
	r_centres = 0.5 * (r_bins[:-1] + r_bins[1:])
	areas = np.pi * (r_bins[1:]**2 - r_bins[:-1]**2)

	# Use last equil_fraction of frames
	n_frames = len(history['positions'])
	start_frame = int(n_frames * (1 - equil_fraction))
	frames = history['positions'][start_frame:]

	profiles = []
	for pos in frames:
		neg_pos = pos[n_pos:]
		neg_r = np.linalg.norm(neg_pos, axis=1)
		counts, _ = np.histogram(neg_r, bins=r_bins)
		profiles.append(counts / areas)

	profiles = np.array(profiles)
	density = np.mean(profiles, axis=0)
	density_std = np.std(profiles, axis=0)
	rho_bg = config.get_rho_bg()

	return r_centres, density, density_std, rho_bg


def measure_effective_mass(history, n_bins=80, equil_fraction=0.5):
	"""
	Compute the effective (deficit) mass as a function of radius.

	M_eff(r) = π ρ_bg r² − M_neg_enc(r)

	This is the mass deficit relative to a uniform background — the quantity
	that acts as "dark matter" in the cavity model.

	Returns
	-------
	r_centres : (n_bins,) radii
	M_eff     : (n_bins,) effective mass at each radius
	M_eff_std : (n_bins,) standard deviation across frames
	"""
	config = history['config']
	n_pos = config.n_positive
	R = config.boundary_radius
	rho_bg = config.get_rho_bg()
	mass_per_particle = abs(config.mass_neg)

	r_eval = np.linspace(0, R, n_bins + 1)[1:]  # skip r=0

	n_frames = len(history['positions'])
	start_frame = int(n_frames * (1 - equil_fraction))
	frames = history['positions'][start_frame:]

	M_eff_all = []
	for pos in frames:
		neg_pos = pos[n_pos:]
		neg_r = np.linalg.norm(neg_pos, axis=1)
		# Enclosed neg mass at each radius
		M_enc = np.array([np.sum(neg_r <= r) * mass_per_particle for r in r_eval])
		# Background expectation
		M_bg = np.pi * rho_bg * r_eval**2 * mass_per_particle
		M_eff_all.append(M_bg - M_enc)

	M_eff_all = np.array(M_eff_all)
	M_eff = np.mean(M_eff_all, axis=0)
	M_eff_std = np.std(M_eff_all, axis=0)

	return r_eval, M_eff, M_eff_std
