"""
Numerical integrators for AntiEngine.

Implements a leapfrog (Störmer-Verlet KDK) integrator — second-order and
symplectic, meaning it conserves a shadow Hamiltonian and remains numerically
stable over long integrations without energy runaway.

KDK leapfrog scheme:
  1. Half-kick:  v += ½ dt · a(x)
  2. Drift:      x += dt · v½
  3. Half-kick:  v½ += ½ dt · a(x_new)
"""

import numpy as np
import jax
import jax.numpy as jnp
from .physics import compute_accelerations


def make_leapfrog_step(masses, G: float, softening: float, dt: float, interaction_mode: str = 'cpt'):
	"""
	Create a JIT-compiled leapfrog integration step.

	masses, G, softening, dt, and interaction_mode are captured as constants in the closure.
	The returned function advances the simulation by one timestep.

	Returns:
		step(positions, velocities) -> (new_positions, new_velocities)
	"""
	@jax.jit
	def step(positions, velocities):
		# Half-kick: accelerate with current forces
		accels = compute_accelerations(positions, masses, G, softening, interaction_mode)
		vel_half = velocities + 0.5 * dt * accels

		# Drift: move with half-kicked velocity
		pos_new = positions + dt * vel_half

		# Half-kick: accelerate with forces at new position
		accels_new = compute_accelerations(pos_new, masses, G, softening, interaction_mode)
		vel_new = vel_half + 0.5 * dt * accels_new

		return pos_new, vel_new

	return step


def make_elastic_boundary_fn(masses, restitution: float = 0.5):
	"""
	Create a JIT-compiled elastic boundary for negative-mass particles.

	When a negative-mass particle moves beyond r_boundary it is reflected off
	the sphere wall — the outward radial velocity is reversed and attenuated by
	the restitution coefficient (1.0 = perfectly elastic, 0.0 = fully absorbing).
	This models neighbouring galactic cavities pressing back ("soap bubble" walls).

	Passing r_boundary as a dynamic argument means the boundary can grow each
	step (expansion mode) without recompiling the JIT kernel.

	Parameters
	----------
	masses      : (N,) JAX array — signed masses; negative particles are confined
	restitution : float in [0, 1] — fraction of outward momentum retained after
	              reflection.  Default 0.5 absorbs half the kinetic energy,
	              avoiding violent rebound shocks while still confining particles.

	Returns
	-------
	apply_boundary(positions, velocities, r_boundary) -> new_velocities
	"""
	neg_mask = masses < 0  # constant boolean mask captured in the closure

	@jax.jit
	def apply_boundary(positions, velocities, r_boundary):
		r3d   = jnp.linalg.norm(positions, axis=1)              # (N,)
		r_hat = positions / jnp.maximum(r3d, 1e-10)[:, None]    # (N, 3)
		v_rad = jnp.sum(velocities * r_hat, axis=1)             # radial speed (N,)

		# Reflect negative-mass particles outside the boundary that are moving outward
		to_reflect = neg_mask & (r3d > r_boundary) & (v_rad > 0)
		new_vel = jnp.where(
			to_reflect[:, None],
			velocities - (1.0 + restitution) * v_rad[:, None] * r_hat,
			velocities,
		)
		return new_vel

	return apply_boundary


def make_pos_boundary_fn(masses):
	"""
	JIT-compiled elastic boundary for positive-mass particles.

	Prevents gas, halo, and other positive-mass particles from escaping the
	simulation domain. Fully elastic (restitution=1): the boundary is just a
	container, not a physical wall. BH (label=5) is already pinned separately.
	"""
	pos_mask = masses > 0

	@jax.jit
	def apply_boundary(positions, velocities, r_boundary):
		r3d   = jnp.linalg.norm(positions, axis=1)
		r_hat = positions / jnp.maximum(r3d, 1e-10)[:, None]
		v_rad = jnp.sum(velocities * r_hat, axis=1)

		to_reflect = pos_mask & (r3d > r_boundary) & (v_rad > 0)
		new_vel = jnp.where(
			to_reflect[:, None],
			velocities - 2.0 * v_rad[:, None] * r_hat,
			velocities,
		)
		return new_vel

	return apply_boundary


def reinject_escaped_particles(
	positions: np.ndarray,
	velocities: np.ndarray,
	neg_offset: int,
	r_boundary: float,
	vel_dispersion: float,
	rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Re-injection boundary: teleport escaped neg-mass particles to the boundary
	surface with inward velocity.

	Models our simulation as a window into an infinite neg-mass background:
	a particle leaving is replaced by a statistically equivalent one entering
	from the opposite side. Prevents shockwaves and boundary pile-up.

	Velocity assignment: isotropic Maxwellian with the radial component
	reflected inward (half-Maxwellian). This is the correct distribution for
	particles crossing a spherical surface inward from a thermal bath.

	Hubble flow is NOT added here — the expanding boundary already models
	cosmological expansion. Adding H₀×r at the boundary would nearly cancel
	the inward drift, causing particles to hover and recycle at the wall.

	Parameters
	----------
	positions      : (N, 3) mutable numpy array [kpc]
	velocities     : (N, 3) mutable numpy array [kpc/Gyr]
	neg_offset     : start index of neg-mass particles in the arrays
	r_boundary     : current boundary radius [kpc]
	vel_dispersion : isotropic velocity dispersion for re-injected particles [kpc/Gyr]
	rng            : numpy random Generator

	Returns
	-------
	(positions, velocities) — modified in-place and returned for convenience
	"""
	neg_pos = positions[neg_offset:]
	neg_vel = velocities[neg_offset:]
	r_neg   = np.linalg.norm(neg_pos, axis=1)

	escaped = r_neg > r_boundary
	n_escaped = int(escaped.sum())
	if n_escaped == 0:
		return positions, velocities

	# Random positions on the boundary surface (uniform on sphere)
	phi   = rng.uniform(0, 2 * np.pi, n_escaped)
	cos_t = rng.uniform(-1, 1, n_escaped)
	sin_t = np.sqrt(1 - cos_t**2)
	new_pos = r_boundary * np.column_stack([
		sin_t * np.cos(phi),
		sin_t * np.sin(phi),
		cos_t,
	])

	# Isotropic thermal velocity with inward-only radial component.
	# Particles entering from a thermal background have a half-Maxwellian
	# in the radial direction (v_r < 0 by definition — they're crossing inward).
	r_hat     = new_pos / r_boundary
	v_thermal = rng.normal(0, vel_dispersion, (n_escaped, 3))
	v_rad     = np.sum(v_thermal * r_hat, axis=1)
	# Reflect any outward radial component inward (preserves tangential components)
	v_thermal -= np.where(v_rad > 0, 2 * v_rad, 0.0)[:, None] * r_hat

	neg_pos[escaped] = new_pos
	neg_vel[escaped] = v_thermal

	return positions, velocities
