"""
Core physics for the AntiEngine particle simulation.

Supports three interaction rule models:

  CPT (default — anti-universe / time-reversed):
    - Positive-positive:  mutual attraction    F =  G|m₁||m₂|/r²
    - Negative-negative:  mutual repulsion     F = -G|m₁||m₂|/r²
    - Positive-negative:  mutual repulsion     F = -G|m₁||m₂|/r²

  Bimetric GR (Petit / Janus model):
    - Same-sign:    mutual attraction (each metric has normal self-gravity)
    - Opposite-sign: mutual repulsion (cross-metric interaction is repulsive)

  Bondi (standard Newton with equivalence principle):
    - a_i = -G m_j / r² r_hat  (acceleration depends only on SOURCE mass sign)
    - Positive mass attracts all; negative mass repels all.
    - pos-neg: asymmetric "runaway" (positive flees, negative chases)

In all cases, inertial mass is |m| (positive). Force magnitude is G|m₁||m₂|/r².
"""

import jax
import jax.numpy as jnp


def _build_interaction_sign(masses, mode='cpt'):
	"""
	Build the (N, N) pairwise interaction sign matrix for the given mode.

	+1 = attractive (acceleration toward source)
	-1 = repulsive (acceleration away from source)
	"""
	sign_i = jnp.sign(masses)[:, None]  # (N, 1)
	sign_j = jnp.sign(masses)[None, :]  # (1, N)

	if mode == 'cpt':
		# Only pos-pos attracts; all other pairs repel
		return jnp.where((sign_i > 0) & (sign_j > 0), 1.0, -1.0)
	elif mode == 'bimetric':
		# Same-sign attracts; opposite-sign repels
		return jnp.where(sign_i * sign_j > 0, 1.0, -1.0)
	elif mode == 'bondi':
		# Acceleration depends only on source mass: a_i = -G m_j/r² r_hat
		# pos source → attract (+1); neg source → repel (-1)
		return jnp.broadcast_to(sign_j, (masses.shape[0], masses.shape[0]))
	else:
		raise ValueError(f"Unknown interaction mode '{mode}'. Choose from: cpt, bimetric, bondi")


def compute_accelerations(positions, masses, G, softening, interaction_mode='cpt'):
	"""
	Compute accelerations for all N particles.

	positions        : (N, D) particle positions (2D or 3D)
	masses           : (N,)   signed masses — positive = normal matter, negative = anti-matter
	G                : gravitational constant (scalar)
	softening        : Plummer softening length, prevents the 1/r singularity at r → 0
	interaction_mode : 'cpt', 'bimetric', or 'bondi'

	Returns (N, D) accelerations.
	"""
	# Displacement vectors r_ij = pos_j - pos_i, shape (N, N, 2)
	r_ij = positions[None, :, :] - positions[:, None, :]

	# Softened |r|^3: (r² + ε²)^(3/2), shape (N, N)
	r2_soft = jnp.sum(r_ij ** 2, axis=-1) + softening ** 2
	r3_inv = r2_soft ** (-1.5)

	interaction_sign = _build_interaction_sign(masses, interaction_mode)

	# Zero out self-interaction (diagonal)
	N = positions.shape[0]
	mask = 1.0 - jnp.eye(N)

	# Acceleration on i due to j = G * |m_j| * sign[i,j] / r³ * r_ij
	# (|m_i| cancels: F_i = G|m_i||m_j|sign/r², a_i = F_i/|m_i| = G|m_j|sign/r²)
	factor = G * jnp.abs(masses)[None, :] * interaction_sign * r3_inv * mask  # (N, N)

	# Sum contributions from all j → (N, D)
	accels = jnp.einsum('ij,ijk->ik', factor, r_ij)

	return accels


def compute_energy(positions, velocities, masses, G, softening, interaction_mode='cpt'):
	"""
	Compute kinetic, potential, and total energy of the system.

	The potential energy follows the interaction convention for the given mode:
	  V_ij = -sign[i,j] * G * |m_i| * |m_j| / r_soft
	where sign is the pairwise interaction sign for the chosen mode.

	For CPT and bimetric this is a valid conservative Hamiltonian.
	For Bondi the asymmetric forces mean this PE is approximate.

	Returns (KE, PE, total_energy) as scalars.
	"""
	abs_masses = jnp.abs(masses)
	N = positions.shape[0]

	# Kinetic energy: ½|m|v², summed over all particles
	KE = 0.5 * jnp.sum(abs_masses * jnp.sum(velocities ** 2, axis=-1))

	# Pairwise softened distances
	r_ij = positions[None, :, :] - positions[:, None, :]
	r2_soft = jnp.sum(r_ij ** 2, axis=-1) + softening ** 2
	r_inv_soft = r2_soft ** (-0.5)

	interaction_sign = _build_interaction_sign(masses, interaction_mode)

	# Sum upper triangle only to avoid double-counting
	upper = jnp.triu(jnp.ones((N, N)), k=1)
	PE = jnp.sum(
		upper * (-interaction_sign) * G
		* abs_masses[:, None] * abs_masses[None, :]
		* r_inv_soft
	)

	return KE, PE, KE + PE
