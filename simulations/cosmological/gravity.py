"""
Particle-Mesh (PM) gravity solver for cosmological simulations.

Uses Cloud-In-Cell (CIC) mass assignment, FFT-based Poisson solver with periodic
boundary conditions, and CIC force interpolation.

Two separate density grids are maintained (positive and negative mass) because
the anti-universe interaction rules produce different force fields for positive
and negative test particles:

  Positive test particle:  a = -∇φ_pos + ∇φ_neg   (attracted to pos, repelled by neg)
  Negative test particle:  a = +∇φ_pos + ∇φ_neg   (repelled by both)

Each φ solves the standard Poisson equation ∇²φ = 4πG·δρ where δρ is the
overdensity (ρ - ρ̄). The mean density drives the Friedmann equation
for the scale factor and is NOT included here — only perturbation forces.
"""

import jax
import jax.numpy as jnp
from functools import partial


# ─────────────────────────────────────────────────────────────────────────────
# CIC mass assignment: particles → grid
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=('n_grid',))
def cic_assign(positions, masses, box_size, n_grid):
	"""
	Cloud-In-Cell mass assignment onto a 3D grid.

	Each particle contributes to the 8 surrounding grid nodes, weighted by
	the trilinear kernel (1 - |dx|/h) in each dimension.

	positions : (N, 3) comoving particle positions in [0, box_size)
	masses    : (N,)   particle masses (absolute values)
	box_size  : comoving box side length
	n_grid    : number of grid cells per dimension

	Returns (n_grid, n_grid, n_grid) density grid in M☉/kpc³.
	"""
	cell_size = box_size / n_grid

	# Particle position in grid units (fractional cell index)
	pos_grid = positions / cell_size

	# Integer cell index of the lower-left corner
	i0 = jnp.floor(pos_grid).astype(jnp.int32)

	# Fractional offset within the cell [0, 1)
	dx = pos_grid - i0

	# Weights for the 8 corners of the CIC cube
	wx = jnp.stack([1.0 - dx[:, 0], dx[:, 0]], axis=-1)  # (N, 2)
	wy = jnp.stack([1.0 - dx[:, 1], dx[:, 1]], axis=-1)
	wz = jnp.stack([1.0 - dx[:, 2], dx[:, 2]], axis=-1)

	grid = jnp.zeros((n_grid, n_grid, n_grid))

	# Loop over the 8 CIC corners
	for di in range(2):
		for dj in range(2):
			for dk in range(2):
				# Periodic wrapping
				ix = (i0[:, 0] + di) % n_grid
				iy = (i0[:, 1] + dj) % n_grid
				iz = (i0[:, 2] + dk) % n_grid
				weight = wx[:, di] * wy[:, dj] * wz[:, dk] * masses
				grid = grid.at[ix, iy, iz].add(weight)

	# Convert mass per cell → density (M☉ / kpc³)
	cell_volume = cell_size ** 3
	return grid / cell_volume


# ─────────────────────────────────────────────────────────────────────────────
# Green's function for the periodic Poisson equation
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=('n_grid',))
def _poisson_greens_function(box_size, n_grid):
	"""
	Precompute the Fourier-space Green's function for ∇²φ = 4πGρ
	on a periodic grid.

	For the standard 7-point finite-difference Laplacian:
	  -k_eff²(k) = (2/h²) × [cos(2πk_x/N) + cos(2πk_y/N) + cos(2πk_z/N) - 3]

	The Green's function is G(k) = -1 / k_eff²(k) for k ≠ 0.
	The k=0 mode is set to zero (mean density doesn't source potential).
	"""
	cell_size = box_size / n_grid
	freqs = jnp.fft.fftfreq(n_grid)  # [-0.5, 0.5) in units of 2π/N

	kx, ky, kz = jnp.meshgrid(freqs, freqs, freqs, indexing='ij')

	# Finite-difference effective k² (matches the discrete Laplacian exactly)
	sin2_x = jnp.sin(jnp.pi * kx) ** 2
	sin2_y = jnp.sin(jnp.pi * ky) ** 2
	sin2_z = jnp.sin(jnp.pi * kz) ** 2
	k_eff_sq = (2.0 / cell_size ** 2) * 2.0 * (sin2_x + sin2_y + sin2_z)

	# Green's function: G = -1/k² (with k=0 zeroed out)
	green = jnp.where(k_eff_sq > 0, -1.0 / k_eff_sq, 0.0)
	return green


# ─────────────────────────────────────────────────────────────────────────────
# Poisson solver: density → potential
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=('n_grid',))
def solve_poisson(density, G_const, box_size, n_grid):
	"""
	Solve ∇²φ = 4πG·δρ via FFT on a periodic grid.

	density   : (n_grid, n_grid, n_grid) overdensity field δρ (M☉/kpc³)
	G_const   : gravitational constant
	box_size  : box side length (kpc)
	n_grid    : grid cells per dimension

	Returns (n_grid, n_grid, n_grid) gravitational potential φ.
	"""
	green = _poisson_greens_function(box_size, n_grid)

	rho_k = jnp.fft.fftn(density)
	phi_k = 4.0 * jnp.pi * G_const * green * rho_k
	phi = jnp.fft.ifftn(phi_k).real

	return phi


# ─────────────────────────────────────────────────────────────────────────────
# Finite-difference gradient: potential → force field
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=('n_grid',))
def gradient_3d(field, box_size, n_grid):
	"""
	Compute the 3D gradient of a periodic scalar field using central differences.

	Returns (3, n_grid, n_grid, n_grid) — gradient components (x, y, z).
	"""
	cell_size = box_size / n_grid
	inv_2h = 1.0 / (2.0 * cell_size)

	# Central differences with periodic wrapping (jnp.roll handles this)
	grad_x = (jnp.roll(field, -1, axis=0) - jnp.roll(field, 1, axis=0)) * inv_2h
	grad_y = (jnp.roll(field, -1, axis=1) - jnp.roll(field, 1, axis=1)) * inv_2h
	grad_z = (jnp.roll(field, -1, axis=2) - jnp.roll(field, 1, axis=2)) * inv_2h

	return jnp.stack([grad_x, grad_y, grad_z], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# CIC force interpolation: grid → particles
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=('n_grid',))
def cic_interpolate(field_3component, positions, box_size, n_grid):
	"""
	Interpolate a 3-component vector field back to particle positions using CIC.

	field_3component : (3, n_grid, n_grid, n_grid) vector field
	positions        : (N, 3) comoving particle positions in [0, box_size)
	box_size         : box side length
	n_grid           : grid cells per dimension

	Returns (N, 3) interpolated field values at particle positions.
	"""
	cell_size = box_size / n_grid
	pos_grid = positions / cell_size
	i0 = jnp.floor(pos_grid).astype(jnp.int32)
	dx = pos_grid - i0

	wx = jnp.stack([1.0 - dx[:, 0], dx[:, 0]], axis=-1)
	wy = jnp.stack([1.0 - dx[:, 1], dx[:, 1]], axis=-1)
	wz = jnp.stack([1.0 - dx[:, 2], dx[:, 2]], axis=-1)

	result = jnp.zeros((positions.shape[0], 3))

	for di in range(2):
		for dj in range(2):
			for dk in range(2):
				ix = (i0[:, 0] + di) % n_grid
				iy = (i0[:, 1] + dj) % n_grid
				iz = (i0[:, 2] + dk) % n_grid
				weight = wx[:, di] * wy[:, dj] * wz[:, dk]  # (N,)

				# Gather field values at this corner for all particles
				for axis in range(3):
					val = field_3component[axis, ix, iy, iz]  # (N,)
					result = result.at[:, axis].add(weight * val)

	return result


# ─────────────────────────────────────────────────────────────────────────────
# Full PM acceleration pipeline
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=('n_grid',))
def pm_accelerations(positions, masses, G_const, box_size, n_grid):
	"""
	Compute PM accelerations for all particles with anti-universe interaction rules.

	The force on each particle depends on its sign:
	  Positive particle: a = -∇φ_pos + ∇φ_neg
	  Negative particle: a = +∇φ_pos + ∇φ_neg

	where φ_pos and φ_neg are potentials sourced by the overdensities of positive
	and negative mass respectively.

	positions : (N, 3) comoving positions in [0, box_size)
	masses    : (N,)   signed masses
	G_const   : gravitational constant
	box_size  : comoving box side length (kpc)
	n_grid    : grid cells per dimension

	Returns (N, 3) comoving accelerations.
	"""
	abs_masses = jnp.abs(masses)

	# Separate positive and negative particles
	pos_mask = masses > 0  # (N,)
	neg_mask = masses < 0

	# CIC mass assignment — two separate grids
	rho_pos = cic_assign(positions, abs_masses * pos_mask, box_size, n_grid)
	rho_neg = cic_assign(positions, abs_masses * neg_mask, box_size, n_grid)

	# Subtract mean density (only perturbations source the potential)
	rho_pos_mean = jnp.mean(rho_pos)
	rho_neg_mean = jnp.mean(rho_neg)
	delta_rho_pos = rho_pos - rho_pos_mean
	delta_rho_neg = rho_neg - rho_neg_mean

	# Solve Poisson for each species
	phi_pos = solve_poisson(delta_rho_pos, G_const, box_size, n_grid)
	phi_neg = solve_poisson(delta_rho_neg, G_const, box_size, n_grid)

	# Gradient → force fields
	grad_phi_pos = gradient_3d(phi_pos, box_size, n_grid)  # (3, ng, ng, ng)
	grad_phi_neg = gradient_3d(phi_neg, box_size, n_grid)

	# Construct force field for positive test particles: -∇φ_pos + ∇φ_neg
	force_field_for_pos = -grad_phi_pos + grad_phi_neg

	# Construct force field for negative test particles: +∇φ_pos + ∇φ_neg
	force_field_for_neg = grad_phi_pos + grad_phi_neg

	# Interpolate forces to particle positions
	accel_from_pos_field = cic_interpolate(force_field_for_pos, positions, box_size, n_grid)
	accel_from_neg_field = cic_interpolate(force_field_for_neg, positions, box_size, n_grid)

	# Select the correct field for each particle based on its sign
	is_pos = (masses > 0)[:, None]  # (N, 1)
	accels = jnp.where(is_pos, accel_from_pos_field, accel_from_neg_field)

	return accels
