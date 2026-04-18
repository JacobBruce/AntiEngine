"""
Phase 2B: Galaxy cluster N-body simulation.

Key observables:
  - Mean pairwise galaxy separation ⟨d_ij⟩ vs time — the dark energy analog.
    If neg-mass pressure dominates gravity, galaxies drift apart.
  - Individual galaxy-center trajectories.
  - Energy conservation as a numerical health check.

Units: kpc, M☉, Gyr.
"""

import numpy as np
import jax.numpy as jnp

from antiengine.physics import compute_accelerations, compute_energy
from antiengine.integrator import make_leapfrog_step, make_elastic_boundary_fn
from simulations.cluster.initial_conditions import (
	ClusterConfig,
	initialize_cluster,
	galaxy_centers,
	print_cluster_ic_summary,
)


# ─────────────────────────────────────────────────────────────────────────────
# Measurement helpers
# ─────────────────────────────────────────────────────────────────────────────

def measure_galaxy_centers(positions, labels, galaxy_ids, n_galaxies):
	"""
	Black-hole position for each galaxy (label=5).

	The BH is by far the most massive particle per galaxy and is the least
	disturbed by inter-galaxy and neg-mass forces, making it the most stable
	proxy for the galaxy's true center of mass. Using CoM would let scattered
	disk particles bias the measurement when the disks are disrupted.

	Falls back to stellar CoM if no BH particle exists for a galaxy.
	Returns (n_galaxies, 3) array of galaxy center positions [kpc].
	"""
	centers = np.zeros((n_galaxies, 3))
	for g in range(n_galaxies):
		bh_mask = (galaxy_ids == g) & (labels == 5)
		if bh_mask.any():
			centers[g] = positions[bh_mask][0]   # BH position
		else:
			# Fallback: mean of all stellar particles for this galaxy
			star_mask = galaxy_ids == g
			if star_mask.any():
				centers[g] = positions[star_mask].mean(axis=0)
	return centers


def measure_pairwise_separations(centers):
	"""
	All unique pairwise separations between galaxy centers.

	Returns a 1D array of length N*(N-1)/2 with distances in kpc,
	ordered as [(0,1), (0,2), ..., (N-2, N-1)].
	"""
	n    = len(centers)
	seps = []
	for i in range(n):
		for j in range(i + 1, n):
			seps.append(float(np.linalg.norm(centers[i] - centers[j])))
	return np.array(seps)


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_cluster_simulation(
	config: ClusterConfig,
	n_steps: int = 1000,
	record_every: int = 20,
) -> dict:
	"""
	Run the galaxy cluster simulation and return a history dict.

	History keys:
	  positions, velocities  : snapshots (list of arrays)
	  time                   : list of float [Gyr]
	  KE, PE, E              : energy snapshots
	  galaxy_centers         : list of (n_galaxies, 3) arrays [kpc]
	  pairwise_separations   : list of 1D arrays [kpc] — one per pair per frame
	  mean_separation        : list of float [kpc]
	  boundary_radius        : list of float [kpc]
	  labels, galaxy_ids, masses, config : static (not per-frame)
	"""
	positions, velocities, masses, labels, galaxy_ids = initialize_cluster(config)
	print_cluster_ic_summary(config)

	print(
		f"\nRunning {n_steps} steps × dt={config.dt} Gyr"
		f" = {n_steps * config.dt:.2f} Gyr"
		f"  (recording every {record_every})"
	)
	print(f"N_total={config.n_total}  softening={config.softening} kpc\n")

	pos_j  = jnp.array(positions)
	vel_j  = jnp.array(velocities)
	mass_j = jnp.array(masses)

	step_fn = make_leapfrog_step(mass_j, config.G, config.softening, config.dt)
	boundary_fn = (
		make_elastic_boundary_fn(mass_j, config.boundary_restitution)
		if config.use_elastic_boundary else None
	)

	r_boundary  = float(config.cluster_radius)
	neg_offset  = config.n_positive  # neg-mass particles start here

	history: dict = {
		'positions':            [],
		'velocities':           [],
		'time':                 [],
		'KE':                   [],
		'PE':                   [],
		'E':                    [],
		'galaxy_centers':       [],
		'pairwise_separations': [],
		'mean_separation':      [],
		'boundary_radius':      [],
		'labels':               labels,
		'galaxy_ids':           galaxy_ids,
		'masses':               masses,
		'config':               config,
	}

	print("JIT compilation complete — starting simulation...")

	t = 0.0
	for i in range(n_steps):
		pos_j, vel_j = step_fn(pos_j, vel_j)

		if boundary_fn is not None:
			vel_j = boundary_fn(pos_j, vel_j, jnp.array(r_boundary))

		# Boundary radius update — two mutually exclusive modes:
		if config.comoving_boundary:
			# Comoving: grow boundary so every galaxy always stays well inside the
			# neg-mass sphere. Use r_max of BH positions (labels==5), not r_rms of
			# all positive particles, so disk-particle noise doesn't affect the result
			# and even the outermost outlier galaxy is guaranteed to be inside.
			# Shell theorem: neg-mass at r > r_galaxy contributes ~zero net force,
			# so as long as all galaxies are well inside the sphere the force on every
			# galaxy is symmetric — exactly the infinite-universe "window" physics.
			bh_mask_np  = labels[:neg_offset] == 5
			bh_radii    = jnp.linalg.norm(pos_j[:neg_offset][bh_mask_np], axis=1)
			r_max_gal   = float(jnp.max(bh_radii)) if bh_radii.size > 0 else float(
				jnp.sqrt(jnp.mean(jnp.sum(pos_j[:neg_offset]**2, axis=1))))
			r_boundary  = max(r_boundary, r_max_gal * config.comoving_safety_factor)
		elif config.use_elastic_boundary and config.boundary_expansion_coupling > 0:
			# Legacy: expand boundary coupled to outward neg-mass velocity
			neg_p  = np.array(pos_j[neg_offset:])
			neg_v  = np.array(vel_j[neg_offset:])
			r_neg  = np.linalg.norm(neg_p, axis=1)
			r_hat  = neg_p / np.maximum(r_neg, 1e-10)[:, None]
			v_rad  = (neg_v * r_hat).sum(axis=1)
			near   = r_neg > 0.85 * r_boundary
			mean_v_out = float(max(0.0, v_rad[near].mean())) if near.any() else 0.0
			r_boundary += config.boundary_expansion_coupling * mean_v_out * config.dt
		r_boundary += config.boundary_expansion_rate * config.dt
		t += config.dt

		if i % record_every == 0:
			pos_np = np.array(pos_j)
			vel_np = np.array(vel_j)

			KE, PE, E = compute_energy(pos_j, vel_j, mass_j, config.G, config.softening)
			gal_centers = measure_galaxy_centers(pos_np, labels, galaxy_ids, config.n_galaxies)
			pair_seps   = measure_pairwise_separations(gal_centers)
			mean_sep    = float(pair_seps.mean()) if len(pair_seps) > 0 else 0.0

			history['positions'].append(pos_np)
			history['velocities'].append(vel_np)
			history['time'].append(t)
			history['KE'].append(float(KE))
			history['PE'].append(float(PE))
			history['E'].append(float(E))
			history['galaxy_centers'].append(gal_centers)
			history['pairwise_separations'].append(pair_seps)
			history['mean_separation'].append(mean_sep)
			history['boundary_radius'].append(r_boundary)

			# Progress line for every recorded frame
			E0    = history['E'][0]
			fluct = abs(float(E) - E0) / abs(E0) * 100 if E0 != 0 else 0.0
			d0    = history['pairwise_separations'][0].mean()
			delta = (mean_sep - d0) / d0 * 100
			if config.comoving_boundary:
				print(f"  t={t:.3f} Gyr  <d>={mean_sep:.1f} kpc"
				      f"  Δ<d>={delta:+.1f}%  r_bnd={r_boundary:.0f} kpc  |ΔE/E₀|={fluct:.3f}%")
			else:
				print(f"  t={t:.3f} Gyr  <d>={mean_sep:.1f} kpc"
				      f"  Δ<d>={delta:+.1f}%  |ΔE/E₀|={fluct:.3f}%")

	n_frames = len(history['positions'])
	print(f"\nDone. {n_frames} frames recorded over {t:.3f} Gyr.")
	return history


def print_cluster_diagnostics(history):
	"""Print a summary of energy conservation and galaxy separation evolution."""
	times     = np.array(history['time'])
	E         = np.array(history['E'])
	E0        = E[0]
	drift     = (E[-1] - E0) / abs(E0) * 100 if E0 != 0 else float('nan')
	max_fluct = np.max(np.abs(E - E0)) / abs(E0) * 100 if E0 != 0 else float('nan')

	mean_seps = np.array(history['mean_separation'])
	d0        = mean_seps[0]
	df        = mean_seps[-1]

	print("\n=== Cluster simulation diagnostics ===")
	print(f"  Duration           : {times[-1]:.3f} Gyr")
	print(f"  Energy drift       : {drift:.4f}%")
	print(f"  Energy max fluct   : {max_fluct:.4f}%")
	print(f"  Mean separation t=0: {d0:.1f} kpc")
	print(f"  Mean separation end: {df:.1f} kpc  ({(df-d0)/d0*100:+.1f}%)")

	# Per-pair breakdown
	pair_seps = history['pairwise_separations']
	config    = history['config']
	n_pairs   = len(pair_seps[0])
	print(f"\n  Per-pair separations (t=0 → t={times[-1]:.2f} Gyr):")
	k = 0
	for i in range(config.n_galaxies):
		for j in range(i + 1, config.n_galaxies):
			d_start = pair_seps[0][k]
			d_end   = pair_seps[-1][k]
			print(f"    Galaxy {i}–{j}: {d_start:.1f} → {d_end:.1f} kpc"
			      f"  ({(d_end - d_start) / d_start * 100:+.1f}%)")
			k += 1
