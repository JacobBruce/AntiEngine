"""
Phase 2B: Galaxy cluster initial conditions.

Places N galaxies at random positions throughout the cluster volume (not on a
surface) so the distribution is isotropic and homogeneous, like the cellular
structure of the cosmic web. Each galaxy carves its own cavity in the shared
neg-mass background.

Each galaxy is a simplified disk + bulge + BH. For pure cluster-scale dynamics
(not individual rotation curves), set n_disk_per_galaxy=0, n_bulge_per_galaxy=0
to collapse each galaxy to a single point mass (BH mass = full galaxy mass).

Units: kpc, M☉, Gyr.
"""

import numpy as np
from dataclasses import dataclass, field
import jax.numpy as jnp

from antiengine.physics import compute_accelerations
from antiengine.units import G_kpc_Msun_Gyr
from simulations.galaxy.initial_conditions import (
	_sample_exponential_disk,
	_sample_hernquist_bulge,
)


@dataclass
class ClusterConfig:
	# ── Galaxy counts ────────────────────────────────────────────────────────
	n_galaxies:         int   = 3     # number of galaxies
	n_disk_per_galaxy:  int   = 150   # disk particles per galaxy
	n_bulge_per_galaxy: int   = 50    # bulge particles per galaxy
	n_bh_per_galaxy:    int   = 1     # BH per galaxy (pinned to galaxy CoM tracking)

	# ── Per-galaxy masses (M☉) ────────────────────────────────────────────
	galaxy_stellar_mass: float = 1.25e11  # total stellar mass per galaxy
	disk_fraction:       float = 0.65     # fraction of stellar mass in disk
	bulge_fraction:      float = 0.35     # fraction of stellar mass in bulge
	bh_mass:             float = 1.4e8    # M☉ — M31* estimate

	# ── Per-galaxy profile parameters (kpc) ──────────────────────────────
	disk_scale_radius:  float = 4.0    # exponential disk r_d
	disk_scale_height:  float = 0.35   # sech² vertical scale z_d
	bulge_scale_radius: float = 1.0    # Hernquist scale radius a

	# ── Cluster layout ────────────────────────────────────────────────────
	# Galaxies are placed uniformly at random throughout the cluster volume
	# with a minimum pairwise separation of galaxy_separation. The placement
	# volume is the cluster_radius sphere (galaxies kept within 0.85 × r_boundary
	# so they don't start too close to the elastic wall).
	# Andromeda is ~750 kpc away, so 600–800 kpc is a realistic minimum separation
	# for a Milky-Way-density region; smaller values allow denser packing.
	galaxy_separation: float = 400.0   # kpc — minimum pairwise galaxy separation

	# ── Particle budget ────────────────────────────────────────────────────
	# If set, distributes a total particle budget evenly across all galaxies,
	# overriding n_disk_per_galaxy and n_bulge_per_galaxy. Useful when scaling
	# n_galaxies so total N stays roughly constant regardless of galaxy count.
	total_star_budget: int | None = None

	# ── Shared negative-mass background ──────────────────────────────────
	n_negative:       int   = 3000    # total neg-mass tracer particles
	neg_mass_ratio:   float = 2.3333  # |total neg mass| / total pos mass
	cluster_radius:   float = 1000.0  # kpc — elastic boundary sphere radius
	neg_inner_radius: float = 20.0    # kpc — exclusion radius per galaxy center

	# ── Neg-mass initial velocity ─────────────────────────────────────────
	neg_vel_dispersion: float = 10.0  # kpc/Gyr Maxwellian σ (0 = start at rest)

	# ── Physics ───────────────────────────────────────────────────────────
	G:         float = G_kpc_Msun_Gyr
	softening: float = 0.5    # kpc — Plummer softening
	dt:        float = 0.002  # Gyr — timestep

	# ── Elastic boundary ──────────────────────────────────────────────────
	use_elastic_boundary:       bool  = True
	boundary_restitution:       float = 0.95
	boundary_expansion_rate:    float = 0.0   # kpc/Gyr fixed linear growth
	boundary_expansion_coupling: float = 1.0  # coupling to mean outward neg-mass velocity
	                                           # (disabled by default when comoving_boundary=True)

	# ── Comoving boundary ─────────────────────────────────────────────────
	# When True, the elastic boundary sphere expands each step to always
	# surround the galaxy distribution: r = max(r, r_rms_galaxies × safety_factor).
	# This ensures all galaxies stay inside the neg-mass sphere regardless of how
	# far they travel, so the cosmological expansion signal is never lost.
	# Neg-mass density naturally decreases as the enclosed volume grows —
	# emergent dilution without any explicit tuning.
	# boundary_expansion_coupling is redundant when this is True.
	comoving_boundary:        bool  = False
	comoving_safety_factor:   float = 2.0   # boundary = max(r_boundary, r_max_BH × factor)
	                                         # 2.0 ensures outermost galaxy is at 50% of boundary
	                                         # — shell theorem then gives symmetric forces

	# ── Hubble-flow initial conditions ──────────────────────────────────
	# Assign each galaxy a bulk radial velocity v = H₀ × r at t=0, where r
	# is its position relative to the cluster center. This eliminates the
	# "warm-up" ramp where galaxies must first be accelerated from rest before
	# developing a Hubble-like flow, letting the simulation start directly on
	# the Hubble reference line. H₀ = 70 km/s/Mpc = 0.07159 Gyr⁻¹ (sim units).
	hubble_flow_ic: bool = False

	# ── Dark energy equation-of-state (CPL parametrization) ─────────────
	# Controls the shape of the Hubble reference curve in the visualization.
	# w(a) = w0 + wa × (1 − a),  a = scale factor = 1/(1+z)
	# ΛCDM: w0 = -1, wa = 0  →  pure de Sitter exponential d₀·exp(H₀·t)
	# DESI DR2 (arXiv:2503.14738, March 2025) best-fit values:
	#
	#   Dataset               Signif  w0      wa     Comment
	#   DESI+CMB+Pantheon+    2.8σ   -0.838  -0.62  Conservative
	#   DESI+CMB+DESY5        4.2σ   -0.752  -0.86  Most significant — default
	#   DESI+CMB (no SNe)     3.1σ   -0.42   -1.75  CMB-only, aggressive
	#
	# The reference curve is obtained by numerically integrating the Friedmann
	# equation with the CPL dark energy density:
	#   ρ_DE(a)/ρ_DE,0 = a^(-3(1+w0+wa)) × exp(-3·wa·(1-a))
	# A decaying dark energy (wa < 0 at w0 > -1) causes the reference curve to
	# fall below the pure ΛCDM exponential at late times.
	w0: float = -0.752   # DESI+CMB+DESY5 4.2σ best fit
	wa: float = -0.86    # DESI+CMB+DESY5 4.2σ best fit

	# ── Randomness ────────────────────────────────────────────────────────
	seed: int = 42

	# ── Derived (computed in __post_init__) ──────────────────────────────
	disk_mass_per_galaxy:   float = field(init=False)
	bulge_mass_per_galaxy:  float = field(init=False)
	total_pos_mass:         float = field(init=False)
	neg_particle_mass:      float = field(init=False)
	n_particles_per_galaxy: int   = field(init=False)
	n_positive:             int   = field(init=False)
	n_total:                int   = field(init=False)

	def __post_init__(self):
		# Auto-scale per-galaxy particle counts from a total budget if given
		if self.total_star_budget is not None:
			self.n_disk_per_galaxy  = max(10, int(self.total_star_budget * self.disk_fraction  / self.n_galaxies))
			self.n_bulge_per_galaxy = max(5,  int(self.total_star_budget * self.bulge_fraction / self.n_galaxies))

		self.disk_mass_per_galaxy  = self.disk_fraction  * self.galaxy_stellar_mass
		self.bulge_mass_per_galaxy = self.bulge_fraction * self.galaxy_stellar_mass
		self.total_pos_mass = self.n_galaxies * (self.bh_mass + self.galaxy_stellar_mass)
		total_neg_mass = self.neg_mass_ratio * self.total_pos_mass
		self.neg_particle_mass      = -total_neg_mass / self.n_negative
		self.n_particles_per_galaxy = (
			self.n_bh_per_galaxy + self.n_disk_per_galaxy + self.n_bulge_per_galaxy
		)
		self.n_positive = self.n_galaxies * self.n_particles_per_galaxy
		self.n_total    = self.n_positive + self.n_negative


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def galaxy_centers(config: ClusterConfig) -> np.ndarray:
	"""
	Place N galaxies at random positions inside the cluster volume.

	Uses rejection sampling: propose a point uniformly at random inside a
	sphere of radius 0.85 × cluster_radius, accept only if the point is at
	least galaxy_separation kpc from every already-accepted galaxy.

	If the packing density is too high (rejection rate stays extreme), falls
	back to Fibonacci sphere so the simulation still runs.

	Returns (n_galaxies, 3) array of galaxy center positions [kpc].
	"""
	N = config.n_galaxies
	if N == 1:
		return np.zeros((1, 3))

	rng     = np.random.default_rng(config.seed)  # deterministic, independent of particle rng
	max_r   = config.cluster_radius * 0.85
	min_sep = config.galaxy_separation

	centers: list = []
	attempts = 0
	max_attempts = 200_000

	while len(centers) < N:
		if attempts >= max_attempts:
			# Packing geometrically infeasible at this density — fall back to shell
			import warnings
			warnings.warn(
				f"Volumetric placement: could not fit {N} galaxies with "
				f"min_sep={min_sep:.0f} kpc in r={max_r:.0f} kpc after {max_attempts} "
				f"attempts. Falling back to Fibonacci sphere. "
				f"Try reducing galaxy_separation or cluster_radius."
			)
			return _fibonacci_sphere_centers(config)
		attempts += 1
		# Uniform point in sphere: r ∝ u^(1/3) gives uniform volume density
		u    = rng.uniform()
		r    = max_r * u ** (1.0 / 3.0)
		cosθ = rng.uniform(-1.0, 1.0)
		sinθ = np.sqrt(max(1.0 - cosθ**2, 0.0))
		phi  = rng.uniform(0.0, 2.0 * np.pi)
		pt   = np.array([r * sinθ * np.cos(phi), r * sinθ * np.sin(phi), r * cosθ])
		# Accept only if far enough from all existing centers
		if all(np.linalg.norm(pt - c) >= min_sep for c in centers):
			centers.append(pt)

	return np.array(centers)


def _fibonacci_sphere_centers(config: ClusterConfig) -> np.ndarray:
	"""Fallback: Fibonacci sphere shell at radius galaxy_separation."""
	N          = config.n_galaxies
	R          = config.galaxy_separation
	phi_golden = (1.0 + np.sqrt(5.0)) / 2.0
	indices    = np.arange(N)
	cos_theta  = 1.0 - 2.0 * (indices + 0.5) / N
	sin_theta  = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))
	phi        = 2.0 * np.pi * indices / phi_golden
	return np.column_stack([
		R * sin_theta * np.cos(phi),
		R * sin_theta * np.sin(phi),
		R * cos_theta,
	])


def _circular_vel_in_plane(pos, accels, center):
	"""
	Assign disk-plane circular orbit velocities relative to `center`.
	v_c = sqrt(|a_radial| * R_xy), direction is tangential in the x-y plane.
	Falls back to a_r ≤ -1e-6 clamp for particles with outward or zero acceleration.
	"""
	dpos = pos - center
	R_xy = np.maximum(np.sqrt(dpos[:, 0]**2 + dpos[:, 1]**2), 1e-3)
	rhat_x = dpos[:, 0] / R_xy
	rhat_y = dpos[:, 1] / R_xy
	# Inward radial component of acceleration
	a_r     = accels[:, 0] * rhat_x + accels[:, 1] * rhat_y
	a_r_neg = np.minimum(a_r, -1e-6)
	v_c     = np.sqrt(np.abs(a_r_neg) * R_xy)
	return np.stack([
		-dpos[:, 1] / R_xy * v_c,
		 dpos[:, 0] / R_xy * v_c,
		 np.zeros(len(pos)),
	], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def initialize_cluster(config: ClusterConfig):
	"""
	Generate initial conditions for the galaxy cluster simulation.

	Particle ordering:
	  For each galaxy g = 0..N-1:
	    [BH (label=5), disk (label=0), bulge (label=1)]
	  Then all neg-mass particles (label=4).

	galaxy_ids encodes which galaxy owns each particle (-1 for neg mass).

	Returns:
	  positions  : (N_total, 3) float64  [kpc]
	  velocities : (N_total, 3) float64  [kpc/Gyr]
	  masses     : (N_total,)   float64  [M☉]  (negative for neg-mass)
	  labels     : (N_total,)   int32    — 0=disk, 1=bulge, 4=neg, 5=BH
	  galaxy_ids : (N_total,)   int32    — galaxy index; -1 for neg-mass
	"""
	rng     = np.random.default_rng(config.seed)
	centers = galaxy_centers(config)
	m_neg   = config.neg_particle_mass

	all_pos, all_vel, all_mass, all_labels, all_galaxy_ids = [], [], [], [], []

	for g_idx, center in enumerate(centers):
		nd = config.n_disk_per_galaxy
		nb = config.n_bulge_per_galaxy

		# ── Sample positions ─────────────────────────────────────────────
		pos_bh = center[None, :].copy()   # BH at galaxy center

		if nd == 0 and nb == 0:
			# ── Point-mass galaxy: BH carries full galaxy mass ────────────
			# No disk or bulge particles — galaxy is a single massive point.
			# bh_mass is set to galaxy_stellar_mass + bh_mass in this case.
			effective_bh_mass = config.galaxy_stellar_mass + config.bh_mass
			all_pos.append(pos_bh)
			all_vel.append(np.zeros((1, 3)))
			all_mass.append(np.array([effective_bh_mass]))
			all_labels.append(np.array([5]))
			all_galaxy_ids.append(np.array([g_idx]))
			continue

		m_disk_per  = config.disk_mass_per_galaxy  / nd if nd > 0 else 0.0
		m_bulge_per = config.bulge_mass_per_galaxy / nb if nb > 0 else 0.0

		pos_components = [pos_bh]
		m_components   = [np.full(1, config.bh_mass)]
		label_parts    = [np.full(1, 5)]

		if nd > 0:
			pos_disk_local, _ = _sample_exponential_disk(
				rng, nd,
				config.disk_scale_radius,
				config.disk_scale_radius * 6,
				config.disk_scale_height,
			)
			pos_components.append(pos_disk_local + center)
			m_components.append(np.full(nd, m_disk_per))
			label_parts.append(np.full(nd, 0))

		if nb > 0:
			pos_bulge_local, _ = _sample_hernquist_bulge(
				rng, nb,
				config.bulge_scale_radius,
				r_max=config.disk_scale_radius * 3,
			)
			pos_components.append(pos_bulge_local + center)
			m_components.append(np.full(nb, m_bulge_per))
			label_parts.append(np.full(nb, 1))

		pos_gal = np.vstack(pos_components)
		m_gal   = np.concatenate(m_components)

		# ── Self-gravity accelerations (galaxy-local only) ────────────────
		accels = np.array(compute_accelerations(
			jnp.array(pos_gal), jnp.array(m_gal),
			config.G, config.softening,
		))

		# ── Circular orbit velocities ────────────────────────────────────
		vel_parts = [np.zeros((1, 3))]  # BH at rest
		offset = 1
		if nd > 0:
			vel_parts.append(_circular_vel_in_plane(
				pos_gal[offset:offset+nd], accels[offset:offset+nd], center))
			offset += nd
		if nb > 0:
			vel_parts.append(_circular_vel_in_plane(
				pos_gal[offset:offset+nb], accels[offset:offset+nb], center))

		# ── Accumulate ───────────────────────────────────────────────────
		all_pos.append(pos_gal)
		all_vel.append(np.vstack(vel_parts))
		all_mass.append(m_gal)
		all_labels.append(np.concatenate(label_parts))
		all_galaxy_ids.append(np.full(1 + nd + nb, g_idx))

	# ── Shared neg-mass background ────────────────────────────────────────
	# Uniformly fill the cluster sphere, excluding within neg_inner_radius
	# of every galaxy center so each galaxy starts with a pre-formed cavity.
	accepted: list = []
	batch = max(config.n_negative * 8, 10000)
	while len(accepted) < config.n_negative:
		# Uniform in sphere of radius cluster_radius
		u    = rng.uniform(0, 1, batch)
		r_s  = config.cluster_radius * u ** (1.0 / 3.0)
		cosθ = rng.uniform(-1, 1, batch)
		sinθ = np.sqrt(1.0 - cosθ**2)
		phi  = rng.uniform(0, 2 * np.pi, batch)
		pts  = np.stack([
			r_s * sinθ * np.cos(phi),
			r_s * sinθ * np.sin(phi),
			r_s * cosθ,
		], axis=1)
		# Reject points too close to any galaxy center
		ok = np.ones(len(pts), dtype=bool)
		for center in centers:
			ok &= np.linalg.norm(pts - center, axis=1) >= config.neg_inner_radius
		accepted.extend(pts[ok].tolist())

	pos_neg = np.array(accepted[:config.n_negative])
	vel_neg = (
		rng.normal(0.0, config.neg_vel_dispersion, (config.n_negative, 3))
		if config.neg_vel_dispersion > 0
		else np.zeros((config.n_negative, 3))
	)

	# ── Stack all components ──────────────────────────────────────────────
	positions  = np.vstack(all_pos + [pos_neg])
	velocities = np.vstack(all_vel + [vel_neg])
	masses     = np.concatenate(all_mass + [np.full(config.n_negative, m_neg)])
	labels     = np.concatenate(all_labels + [np.full(config.n_negative, 4)])
	galaxy_ids = np.concatenate(all_galaxy_ids + [np.full(config.n_negative, -1)])

	# ── Hubble-flow bulk velocities ──────────────────────────────────────
	# Adds v = H₀ × r_galaxy to all particles belonging to each galaxy so
	# the cluster starts already in Hubble recession, not from rest.
	if config.hubble_flow_ic:
		H0 = 0.07159  # Gyr⁻¹  (70 km/s/Mpc in simulation units)
		for g_idx, center in enumerate(centers):
			mask = galaxy_ids == g_idx
			velocities[mask] += H0 * center  # v_hubble = H₀ × r_vec

	return (
		positions.astype(np.float64),
		velocities.astype(np.float64),
		masses.astype(np.float64),
		labels.astype(np.int32),
		galaxy_ids.astype(np.int32),
	)


def print_cluster_ic_summary(config: ClusterConfig):
	"""Print a concise summary of cluster initial conditions."""
	centers = galaxy_centers(config)
	seps = [
		np.linalg.norm(centers[i] - centers[j])
		for i in range(config.n_galaxies)
		for j in range(i + 1, config.n_galaxies)
	]
	sep_mean = float(np.mean(seps)) if seps else 0.0
	sep_min  = float(np.min(seps))  if seps else 0.0
	sep_max  = float(np.max(seps))  if seps else 0.0

	point_mass = (config.n_disk_per_galaxy == 0 and config.n_bulge_per_galaxy == 0)
	eff_bh_mass = (config.galaxy_stellar_mass + config.bh_mass) if point_mass else config.bh_mass

	print("\n=== Cluster initial conditions ===")
	print(f"  Galaxies       : {config.n_galaxies}  (volumetric random, min sep={config.galaxy_separation:.0f} kpc)")
	print(f"  Pairwise seps  : {sep_mean:.0f} kpc mean  ({sep_min:.0f}–{sep_max:.0f} kpc range)")
	print(f"  Per galaxy     : {config.n_bh_per_galaxy} BH + {config.n_disk_per_galaxy} disk"
	      f" + {config.n_bulge_per_galaxy} bulge = {config.n_particles_per_galaxy} particles")
	if point_mass:
		print(f"                   M_bh={eff_bh_mass:.2e}  (point mass — full galaxy mass in BH)  M☉")
	else:
		print(f"                   M_bh={eff_bh_mass:.2e}  M_disk={config.disk_mass_per_galaxy:.2e}"
		      f"  M_bulge={config.bulge_mass_per_galaxy:.2e}  M☉")
	print(f"  Neg. mass      : {config.n_negative} particles  (ratio={config.neg_mass_ratio}x,"
	      f"  r_boundary={config.cluster_radius:.0f} kpc)")
	print(f"  Total N        : {config.n_total}")
	print(f"  M_neg / M_pos  : {config.neg_mass_ratio:.1f}×")
	if config.hubble_flow_ic:
		H0 = 0.07159
		print(f"  Hubble-flow IC : enabled  (H₀×r added to each galaxy, H₀={H0} Gyr⁻¹)")
	else:
		print(f"  Hubble-flow IC : disabled  (galaxies start from rest)")
