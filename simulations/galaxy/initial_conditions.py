"""
Initial conditions for the galactic-scale simulation.

All quantities are in simulation units: kpc, M☉, Gyr.

Galaxy components modelled:
  - Stellar disk   : exponential surface density  Σ(R) ∝ exp(-R / r_d)
                     Thin sech² vertical profile   ρ(z) ∝ sech²(z / 2z_d)
  - Stellar bulge  : Hernquist profile             ρ(r) ∝ 1 / (r/a)(1 + r/a)³
  - Stellar halo   : NFW-like power law            ρ(r) ∝ 1 / r  (truncated)
  - Gas halo       : Beta-model                    ρ(r) ∝ (1 + r²/r_c²)^(-3β/2)
  - Negative mass  : Uniform sphere of radius r_neg (the "sea" of anti-matter)

Orbital velocities:
  Each particle type is placed with the circular orbit velocity
  v_c(R) = sqrt(G * M_enc(R) / R)  derived from the total enclosed mass at
  its birth radius. For disk/bulge particles, the enclosed mass includes
  contributions from ALL positive-mass components, computed analytically.

  This ensures the galaxy is in approximate dynamical equilibrium at t=0 and
  that any rotation-curve evolution we measure is due to the negative-mass
  cavity forming, not numerical transients.
"""

import numpy as np
from dataclasses import dataclass, field
import jax.numpy as jnp

from antiengine.physics import compute_accelerations
from antiengine.units import (
	G_kpc_Msun_Gyr,
	kms_to_kpc_per_Gyr,
	AND_STELLAR_MASS_MSUN,
	AND_BULGE_FRACTION,
	AND_DISK_FRACTION,
	AND_HALO_FRACTION,
	AND_DISK_SCALE_RADIUS_KPC,
	AND_BULGE_SCALE_RADIUS_KPC,
	AND_STELLAR_HALO_RADIUS_KPC,
	AND_GAS_HALO_RADIUS_KPC,
	GAS_TO_STELLAR_FRACTION,
)


@dataclass
class GalaxyConfig:
	# ── Particle counts ─────────────────────────────────────────────────────
	n_disk:     int   = 2000   # stellar disk particles
	n_bulge:    int   = 500    # stellar bulge particles
	n_halo:     int   = 500    # stellar halo particles
	n_gas:      int   = 500    # gas halo particles
	n_negative: int   = 3000   # negative-mass background particles

	# ── Stellar masses (total, M☉) ────────────────────────────────────────
	total_stellar_mass: float = AND_STELLAR_MASS_MSUN   # M☉
	disk_fraction:      float = AND_DISK_FRACTION
	bulge_fraction:     float = AND_BULGE_FRACTION
	halo_fraction:      float = AND_HALO_FRACTION

	# ── Profile parameters (kpc) ──────────────────────────────────────────
	disk_scale_radius:        float = AND_DISK_SCALE_RADIUS_KPC        # r_d
	disk_scale_height:        float = 0.35                              # z_d (kpc)
	bulge_scale_radius:       float = AND_BULGE_SCALE_RADIUS_KPC       # Hernquist a
	stellar_halo_radius:       float = AND_STELLAR_HALO_RADIUS_KPC     # outer truncation
	stellar_halo_power:        float = 1.5                              # density slope ρ ∝ r^{-n}
	stellar_halo_inner_radius: float = 8.0                             # inner exclusion (kpc) — must be outside disk
	gas_halo_radius:          float = AND_GAS_HALO_RADIUS_KPC
	gas_halo_beta:            float = 0.5                               # beta model
	gas_core_radius:          float = 5.0                               # r_c (kpc)

	# ── Central black hole ──────────────────────────────────────────────
	n_bh:    int   = 1        # always 1 — central BH (pinned to origin throughout)
	bh_mass: float = 140e6   # M☉ — M31* 2005 estimate midpoint (~110–230 ×10⁶ M☉)

	# ── Negative mass ─────────────────────────────────────────────────────
	neg_mass_ratio:      float = 9.0    # |total neg mass| / total pos mass
	neg_sphere_radius:   float = 150.0  # kpc — outer edge of negative mass region
	neg_inner_radius:    float = 20.0   # kpc — inner exclusion (beyond 5×r_d disk edge)

	# ── Simulation parameters ─────────────────────────────────────────────
	G:                       float = G_kpc_Msun_Gyr
	softening:               float = 0.5     # kpc — Plummer softening
	dt:                      float = 0.002   # Gyr — timestep (~2 Myr)
	use_elastic_boundary:    bool  = True    # confine neg mass within neg_sphere_radius
	use_reinjection_boundary: bool = False   # teleport escaped neg-mass to boundary surface
	boundary_restitution:         float = 0.95   # 0=absorbing wall  1=perfectly elastic
	boundary_expansion_rate:      float = 0.0   # kpc/Gyr — fixed linear growth of bubble
	boundary_expansion_coupling:  float = 1.0   # dimensionless — couples wall growth to mean outward velocity of near-wall neg particles
	hubble_expansion_boundary:    bool  = False  # expand boundary at Hubble rate
	hubble_rate:                  float = 0.07159  # H₀ in Gyr⁻¹ (70 km/s/Mpc)

	# ── Background density → auto-derived sphere radius ────────────────────
	# Target neg-mass background density (M☉/kpc³). When set > 0, neg_sphere_radius
	# is auto-derived in __post_init__ so actual particle density matches this value:
	#   r = (3 × M_neg / (4π × ρ))^(1/3)
	#
	# This makes M_eff measurements self-consistent: the N/V density in the
	# simulation equals the target cosmic density, no override needed.
	# If 0: neg_sphere_radius is used as-is (manual setting).
	#
	# Physically motivated values (sim units: M☉/kpc³):
	#   Cosmic critical density:        ~136 M☉/kpc³
	#   Cosmic average (50/50 model):     ~7 M☉/kpc³  (far too low for DM halos)
	#   MW DM halo (M≈1.3e12 within 200 kpc): ~3.9e4 M☉/kpc³  (~285× ρ_crit)
	#   ΛCDM r_200 overdensity:        ~2.7e4 M☉/kpc³  (200× ρ_crit by definition)
	#
	# The cavity model requires LOCAL neg-mass overdensity around galaxies,
	# comparable to ΛCDM dark matter halo overdensities (~200-300× ρ_crit).
	# Cosmic average alone is ~5800× too low.
	neg_bg_density: float = 0.0  # M☉/kpc³

	# Use r_boundary as cavity radius instead of the measured gradient-peak value.
	# The boundary IS the true cavity edge — the measured value is an approximation
	# that tracks slightly inside due to the smoothed neg-mass density transition.
	cavity_from_boundary: bool = False

	# ── Negative mass initial velocity ─────────────────────────────────────
	# Isotropic Maxwellian velocity dispersion for neg-mass particles at t=0.
	# A small thermal speed (σ≈10 kpc/Gyr ≈ 10 km/s) spreads the phase of
	# boundary reflections, preventing coherent pile-up shockwaves.
	neg_vel_dispersion: float = 10.0   # kpc/Gyr  (0 = start at rest)

	# Dark matter density profile exponent for the effective rotation curve.
	# The cavity dark mass is modelled as ρ_DM(r) ∝ r^(−neg_density_slope) inside the cavity.
	# Enclosed dark mass: M_enc(r) = M_total × (r/r_cav)^(3−slope).
	# Higher values concentrate the dark mass toward the center of the galaxy:
	#   0   : uniform density → M_enc ∝ r³  → v_c rises linearly (solid body, no uplift)
	#   1   : isothermal      → M_enc ∝ r²  → v_c = const (perfectly flat)
	#   1.5 : NFW-like        → M_enc ∝ r^1.5 → v_c gently declining (most realistic)
	#   2   : steeper         → M_enc ∝ r   → v_c ∝ r^{−0.5}
	# Uplift extends smoothly to r=0 (no hard cutoff at neg_inner_radius).
	neg_density_slope: float = 1.5

	# ── Randomness ────────────────────────────────────────────────────────
	seed: int = 42

	# ── Interaction rules ─────────────────────────────────────────────────
	# 'cpt'      : pos-pos attract, neg-neg repel, pos-neg mutual repulsion
	# 'bimetric' : same-sign attract, opposite-sign repel
	# 'bondi'    : a = -Gm_j/r² (runaway: neg chases pos, pos flees neg)
	interaction_mode: str = 'cpt'

	# Derived (computed post-init)
	disk_mass:     float = field(init=False)
	bulge_mass:    float = field(init=False)
	halo_mass:     float = field(init=False)
	gas_mass:      float = field(init=False)
	total_pos_mass: float = field(init=False)
	neg_particle_mass: float = field(init=False)

	def __post_init__(self):
		self.disk_mass  = self.total_stellar_mass * self.disk_fraction
		self.bulge_mass = self.total_stellar_mass * self.bulge_fraction
		self.halo_mass  = self.total_stellar_mass * self.halo_fraction
		self.gas_mass   = self.total_stellar_mass * GAS_TO_STELLAR_FRACTION
		self.total_pos_mass = (
			self.disk_mass + self.bulge_mass + self.halo_mass + self.gas_mass + self.bh_mass
		)
		total_neg_mass = self.total_pos_mass * self.neg_mass_ratio
		self.neg_particle_mass = -total_neg_mass / self.n_negative

		# Auto-derive neg_sphere_radius from neg_bg_density so the actual
		# particle density matches the target cosmic density exactly.
		# r = (3 × M_neg / (4π × ρ))^(1/3)
		if self.neg_bg_density > 0:
			self.neg_sphere_radius = (
				3 * total_neg_mass / (4 * np.pi * self.neg_bg_density)
			) ** (1.0 / 3.0)


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_exponential_disk(rng, n, r_d, r_max, z_d):
	"""
	Sample cylindrical coords (R, phi, z) from an exponential disk.
	  Σ(R) ∝ exp(-R / r_d)  sampled by inverse CDF (R) and accept-reject.
	  z follows a sech² distribution, approximated by double-exponential.
	"""
	# Radial: inverse CDF of 2π R exp(-R/r_d)  using rejection sampling
	Rs = []
	while len(Rs) < n:
		R_cand = rng.exponential(scale=r_d, size=(n * 4,))
		# Weight for the R factor: pdf ∝ R exp(-R/r_d), exponential gives exp(-R/r_d)
		# so we need to weight by R/r_d (normalised to max=1)
		weight = R_cand / (r_d * np.e)           # max of R/r_d * e^(-1) is at R=r_d
		accept = rng.random(len(R_cand)) < weight
		Rs.extend(R_cand[accept][:n - len(Rs)])
	R   = np.array(Rs[:n])
	phi = rng.uniform(0, 2 * np.pi, n)
	# Vertical: sech² ≈ double-exponential with scale z_d
	z   = rng.laplace(scale=z_d, size=n)

	x = R * np.cos(phi)
	y = R * np.sin(phi)
	return np.stack([x, y, z], axis=1), R


def _sample_hernquist_bulge(rng, n, a, r_max=None, r_min=0.0):
	"""
	Sample from a Hernquist profile: ρ(r) ∝ 1 / (r/a)(1 + r/a)³.
	CDF: M(<r) = M * r² / (r + a)²  → inverse CDF: r = a sqrt(u) / (1 - sqrt(u))
	Clamp to [r_min, r_max] by restricting the CDF range to [u_min, u_max].
	r_min enforces an inner exclusion zone (useful for halos that shouldn't overlap the bulge).
	"""
	if r_max is None:
		r_max = a * 100.0
	u_max = (r_max / (r_max + a)) ** 2
	u_min = (r_min / (r_min + a)) ** 2 if r_min > 0 else 0.0
	u     = rng.uniform(u_min, u_max, n)
	r     = a * np.sqrt(u) / (1.0 - np.sqrt(u))
	# Isotropic: uniform on sphere
	cosθ = rng.uniform(-1, 1, n)
	phi  = rng.uniform(0, 2 * np.pi, n)
	sinθ = np.sqrt(1 - cosθ**2)
	x = r * sinθ * np.cos(phi)
	y = r * sinθ * np.sin(phi)
	z = r * cosθ
	return np.stack([x, y, z], axis=1), r


def _sample_power_law_sphere(rng, n, r_max, power, r_min=0.0):
	"""
	Sample from ρ ∝ r^(-power) in a sphere, optionally excluding r < r_min.

	For power < 3: closed-form inverse CDF with inner exclusion.
	  CDF: M(<r) ∝ r^(3-power)  →  r = (r_min^(3-p) + u*(r_max^(3-p) - r_min^(3-p)))^(1/(3-p))
	For power ≥ 3: rejection sampling (r_min treated as core radius if given).
	"""
	if power >= 3.0:
		r_core = max(r_min, 0.01 * r_max)
		r_list = []
		while len(r_list) < n:
			u = rng.uniform(0, 1, n * 10)
			r_cand = r_core / (1.0 - u * (1.0 - r_core / r_max))
			r_cand = np.clip(r_cand, r_core, r_max)
			accept_prob = np.clip((r_cand / r_core) ** (2.0 - power), 0.0, 1.0)
			accept = rng.random(len(r_cand)) < accept_prob
			r_list.extend(r_cand[accept].tolist())
		r = np.array(r_list[:n])
	else:
		e  = 3.0 - power   # exponent; > 0 for power < 3
		lo = r_min ** e if r_min > 0 else 0.0
		hi = r_max ** e
		u  = rng.uniform(0, 1, n)
		r  = (lo + u * (hi - lo)) ** (1.0 / e)
	cosθ = rng.uniform(-1, 1, n)
	phi  = rng.uniform(0, 2 * np.pi, n)
	sinθ = np.sqrt(1 - cosθ**2)
	x = r * sinθ * np.cos(phi)
	y = r * sinθ * np.sin(phi)
	z = r * cosθ
	return np.stack([x, y, z], axis=1), r


def _sample_beta_sphere(rng, n, r_c, r_max, beta):
	"""
	Sample from the beta-model: ρ ∝ (1 + (r/r_c)²)^(-3β/2)
	via rejection sampling.
	"""
	alpha = 3.0 * beta / 2.0
	pos_list = []
	while len(pos_list) < n:
		# Uniform in sphere of r_max
		u_r   = rng.uniform(0, 1, n * 10)
		r_cand = r_max * u_r ** (1.0 / 3.0)
		# Probability weight  ∝  (1 + (r/r_c)²)^(-alpha) / r_max^(-alpha)
		# normalised by max at r=0 which equals 1
		weight = (1 + (r_cand / r_c) ** 2) ** (-alpha)
		accept = rng.random(len(r_cand)) < weight
		pos_list.extend(r_cand[accept].tolist())
	r    = np.array(pos_list[:n])
	cosθ = rng.uniform(-1, 1, n)
	phi  = rng.uniform(0, 2 * np.pi, n)
	sinθ = np.sqrt(1 - cosθ**2)
	x = r * sinθ * np.cos(phi)
	y = r * sinθ * np.sin(phi)
	z = r * cosθ
	return np.stack([x, y, z], axis=1), r


def _sample_uniform_sphere(rng, n, r_max):
	"""Uniform distribution inside a sphere of radius r_max."""
	r    = r_max * rng.uniform(0, 1, n) ** (1.0 / 3.0)
	cosθ = rng.uniform(-1, 1, n)
	phi  = rng.uniform(0, 2 * np.pi, n)
	sinθ = np.sqrt(1 - cosθ**2)
	x = r * sinθ * np.cos(phi)
	y = r * sinθ * np.sin(phi)
	z = r * cosθ
	return np.stack([x, y, z], axis=1), r


# ─────────────────────────────────────────────────────────────────────────────
# Enclosed mass (analytical) — used to compute circular orbit velocities
# ─────────────────────────────────────────────────────────────────────────────

def _M_enc_disk(R, M_disk, r_d):
	"""
	Enclosed mass of an infinitely thin exponential disk within radius R
	(spherical approximation — underestimates slightly but adequate for orbit init).
	Exact cylindrical form: M(<R) = M_disk * [1 - exp(-R/r_d)(1 + R/r_d)]
	"""
	x = R / r_d
	return M_disk * (1.0 - np.exp(-x) * (1.0 + x))


def _M_enc_hernquist(r, M_bulge, a):
	"""Hernquist enclosed mass: M(<r) = M_bulge * r² / (r + a)²"""
	return M_bulge * r**2 / (r + a)**2


def _M_enc_power_sphere(r, M_halo, r_max, power):
	"""Power-law sphere enclosed mass: M(<r) = M_halo * (r/r_max)^(3-power)"""
	return M_halo * (np.clip(r, 0, r_max) / r_max) ** (3.0 - power)


def _M_enc_uniform_sphere(r, M_total, r_max):
	"""Uniform sphere enclosed mass: M(<r) = M_total * (r/r_max)³"""
	return M_total * (np.clip(r, 0, r_max) / r_max) ** 3


def _circular_velocity(r, M_enc_total, G):
	"""v_c = sqrt(G * M_enc / r); safe for r=0."""
	r_safe = np.maximum(r, 1e-3)
	return np.sqrt(G * M_enc_total / r_safe)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def initialize_galaxy(config: GalaxyConfig):
	"""
	Generate initial conditions for the single-galaxy simulation.

	Returns:
		positions  : (N, 3) float64 array  [kpc]
		velocities : (N, 3) float64 array  [kpc/Gyr]
		masses     : (N,)   float64 array  [M☉]  (signed; negative for anti-matter)
		labels     : (N,)   int array  —  0=disk, 1=bulge, 2=stellar_halo,
		                                  3=gas, 4=negative
	"""
	rng = np.random.default_rng(config.seed)
	G   = config.G

	# ── Per-particle masses ───────────────────────────────────────────────
	m_disk   = config.disk_mass   / config.n_disk
	m_bulge  = config.bulge_mass  / config.n_bulge
	m_halo   = config.halo_mass   / config.n_halo
	m_gas    = config.gas_mass    / config.n_gas
	m_neg    = config.neg_particle_mass   # already negative

	# ── Sample positions ──────────────────────────────────────────────────
	pos_disk,  R_disk  = _sample_exponential_disk(
		rng, config.n_disk,
		config.disk_scale_radius, config.disk_scale_radius * 6,
		config.disk_scale_height,
	)
	pos_bulge, r_bulge = _sample_hernquist_bulge(
		rng, config.n_bulge, config.bulge_scale_radius,
		r_max=config.disk_scale_radius * 3,  # truncate bulge at ~3 r_d
	)
	# Power-law halo: ρ ∝ r^{-power} from r_min to r_max.
	# Moderate slope (default power=1.5) spreads particles across 8–34 kpc,
	# well outside the disk+bulge region. Closed-form inverse CDF with inner exclusion.
	pos_shalo, r_shalo = _sample_power_law_sphere(
		rng, config.n_halo,
		r_max=config.stellar_halo_radius,
		power=config.stellar_halo_power,
		r_min=config.stellar_halo_inner_radius,
	)
	pos_gas,   r_gas   = _sample_beta_sphere(
		rng, config.n_gas,
		config.gas_core_radius, config.gas_halo_radius, config.gas_halo_beta,
	)
	# ── Central black hole — fixed point mass at the origin ──────────────────
	# The BH is pinned to (0,0,0) throughout the simulation (velocity zeroed each
	# step in the run loop).  Its mass provides a stable gravitational anchor
	# for the inner bulge without contributing to N-body drift.
	pos_bh = np.zeros((config.n_bh, 3))
	vel_bh = np.zeros((config.n_bh, 3))

	# Neg mass: shell sampling with BOTH spherical and cylindrical exclusion.
	# Spherical: r >= neg_inner_radius (20 kpc) — keeps neg mass out of bulge volume.
	# Cylindrical: R_xy >= stellar_halo_inner_radius (8 kpc) — prevents polar-cap
	#   particles (cosθ ≈ ±1) from projecting onto the disk center in the x-y view.
	#   These particles have (x≈0, y≈0, z≈±r) in 3D but appear at (0,0) in projection,
	#   falsely suggesting they sit on top of the bulge and BH.
	r3_lo       = config.neg_inner_radius ** 3
	r3_hi       = config.neg_sphere_radius ** 3
	R_min_cyl   = config.stellar_halo_inner_radius   # 8 kpc — disk-plane exclusion
	accepted: list = []
	batch = config.n_negative * 6   # generous batch for fast convergence
	while len(accepted) < config.n_negative:
		r_s    = (r3_lo + rng.uniform(0, 1, batch) * (r3_hi - r3_lo)) ** (1.0 / 3.0)
		cosθ   = rng.uniform(-1, 1, batch)
		sinθ   = np.sqrt(1.0 - cosθ**2)
		phi    = rng.uniform(0, 2 * np.pi, batch)
		R_xy   = r_s * sinθ
		ok     = R_xy >= R_min_cyl
		rows   = np.stack([R_xy * np.cos(phi), R_xy * np.sin(phi), r_s * cosθ], axis=1)
		accepted.extend(rows[ok].tolist())
	pos_neg   = np.array(accepted[: config.n_negative])
	r_neg     = np.linalg.norm(pos_neg, axis=1)

	# ── Compute circular orbit velocities using the stellar-only N-body force ─
	# We assemble only the positive-mass components to compute the centripetal
	# acceleration for each positive-mass particle.  Excluding neg-mass from the
	# force field here ensures that:
	#   a) outer disk particles (r ≈ 20–24 kpc) aren't given near-zero velocity
	#      because a handful of neg-mass particles near the inner exclusion radius
	#      locally reduce or reverse the centripetal force;
	#   b) the rotation curve is independent of neg_mass_ratio at t=0, which is
	#      the correct baseline — the cavity effect develops dynamically.
	all_pos_stellar = np.vstack([pos_bh, pos_disk, pos_bulge, pos_shalo, pos_gas])
	all_masses_stellar = np.concatenate([
		np.full(config.n_bh,    config.bh_mass),
		np.full(config.n_disk,  m_disk),
		np.full(config.n_bulge, m_bulge),
		np.full(config.n_halo,  m_halo),
		np.full(config.n_gas,   m_gas),
	])
	# Compute stellar-only accelerations (no neg mass)
	accels_all = np.array(compute_accelerations(
		jnp.array(all_pos_stellar),
		jnp.array(all_masses_stellar),
		config.G, config.softening,
	))

	def circular_vel_from_force(pos, accels, n_end):
		"""
		Set circular orbit velocity from actual N-body centripetal acceleration.
		v_c(i) = sqrt(|a_radial(i)| * R_xy(i))
		Direction is the tangential unit vector in the x-y plane.
		"""
		R_xy = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
		R_xy = np.maximum(R_xy, 1e-3)
		# Radial unit vector in x-y plane
		rhat_x = pos[:, 0] / R_xy
		rhat_y = pos[:, 1] / R_xy
		# Inward radial acceleration component (x-y plane only)
		a_r = accels[:, 0] * rhat_x + accels[:, 1] * rhat_y
		# For a stable orbit we need centripetal force inward: a_r < 0
		# If a_r > 0 (particle pushed outward, e.g. by neg mass), use analytical fallback
		a_r_neg = np.minimum(a_r, -1e-6)  # clamp to at most -1e-6 kpc/Gyr²
		v_c = np.sqrt(np.abs(a_r_neg) * R_xy)
		vel = np.stack([
			-pos[:, 1] / R_xy * v_c,
			 pos[:, 0] / R_xy * v_c,
			 np.zeros(len(pos)),
		], axis=1)
		return vel

	def dispersion_vel_from_force(rng_local, pos, accels):
		"""
		Isotropic velocity dispersion from the local N-body force.
		Used for the gas halo (genuinely pressure-supported, far from center).
		σ_1D = sqrt(|a| * r_3D) / sqrt(3) → 3D Maxwellian.
		"""
		r_3d  = np.maximum(np.linalg.norm(pos, axis=1), 1e-3)
		a_mag = np.linalg.norm(accels, axis=1)
		sigma = np.sqrt(a_mag * r_3d) / np.sqrt(3.0)
		vel = np.column_stack([
			rng_local.normal(0.0, sigma),
			rng_local.normal(0.0, sigma),
			rng_local.normal(0.0, sigma),
		])
		return vel

	def spherical_circ_vel_from_force(rng_local, pos, accels):
		"""
		Circular orbits with randomly oriented planes for the stellar halo.

		Each particle's speed is the circular velocity from the N-body force,
		but the orbital plane is tilted at a random angle in 3D.  This gives a
		hot, isotropic-looking velocity field in projection while guaranteeing
		that no particle plunges through the center (pericenter = current radius).

		This prevents stellar halo particles from disrupting the dense bulge
		region — the core problem with full Maxwellian dispersion at low N.
		"""
		r_3d  = np.maximum(np.linalg.norm(pos, axis=1), 1e-3)
		r_hat = pos / r_3d[:, None]
		# Inward radial component of full 3D acceleration
		a_r = np.sum(accels * r_hat, axis=1)
		a_r_inward = np.minimum(a_r, -1e-6)
		v_c = np.sqrt(np.abs(a_r_inward) * r_3d)
		# Random unit vector tangential to r (perpendicular to r_hat):
		# Draw a random vector, remove the radial projection, normalise.
		u  = rng_local.normal(0.0, 1.0, pos.shape)
		u -= np.sum(u * r_hat, axis=1)[:, None] * r_hat   # Gram-Schmidt
		u /= np.maximum(np.linalg.norm(u, axis=1)[:, None], 1e-10)
		return u * v_c[:, None]

	n_bh = config.n_bh   # offset: BH is first
	n_d  = config.n_disk
	n_b  = config.n_bulge
	n_sh = config.n_halo
	n_g  = config.n_gas

	# Slice out per-component accelerations (skip BH — it is pinned, not orbiting)
	accels_disk  = accels_all[n_bh             : n_bh + n_d]
	accels_bulge = accels_all[n_bh + n_d       : n_bh + n_d + n_b]
	accels_shalo = accels_all[n_bh + n_d + n_b : n_bh + n_d + n_b + n_sh]
	accels_gas   = accels_all[n_bh + n_d + n_b + n_sh : n_bh + n_d + n_b + n_sh + n_g]

	vel_disk  = circular_vel_from_force(pos_disk,  accels_disk,  n_d)
	vel_bulge = circular_vel_from_force(pos_bulge, accels_bulge, n_b)
	# Halo: circular orbits with random 3D tilt — stable but isotropic in projection
	vel_shalo = spherical_circ_vel_from_force(rng, pos_shalo, accels_shalo)
	# Gas halo: pressure-supported, genuinely Maxwellian (far from center, no disruption risk)
	vel_gas   = dispersion_vel_from_force(rng, pos_gas, accels_gas)

	# Negative mass particles: initial isotropic velocity dispersion.
	# A small thermal speed spreads boundary-reflection phases, reducing the
	# pile-up shockwave that forms when all particles bounce back in sync.
	if config.neg_vel_dispersion > 0:
		vel_neg = rng.normal(0.0, config.neg_vel_dispersion, (config.n_negative, 3))
	else:
		vel_neg = np.zeros((config.n_negative, 3))

	# ── Stack all components (BH first for stable indexing) ───────────────────
	positions  = np.vstack([pos_bh, pos_disk, pos_bulge, pos_shalo, pos_gas, pos_neg])
	velocities = np.vstack([vel_bh, vel_disk, vel_bulge, vel_shalo, vel_gas, vel_neg])
	masses     = np.concatenate([
		np.full(config.n_bh,       config.bh_mass),
		np.full(config.n_disk,     m_disk),
		np.full(config.n_bulge,    m_bulge),
		np.full(config.n_halo,     m_halo),
		np.full(config.n_gas,      m_gas),
		np.full(config.n_negative, m_neg),
	])
	labels = np.concatenate([
		np.full(config.n_bh,       5),   # black hole
		np.full(config.n_disk,     0),   # disk
		np.full(config.n_bulge,    1),   # bulge
		np.full(config.n_halo,     2),   # stellar halo
		np.full(config.n_gas,      3),   # gas
		np.full(config.n_negative, 4),   # negative mass
	])

	return (
		positions.astype(np.float64),
		velocities.astype(np.float64),
		masses.astype(np.float64),
		labels.astype(np.int32),
	)


def print_ic_summary(config: GalaxyConfig):
	"""Print a human-readable summary of the initial conditions setup."""
	print("=== Galaxy initial conditions ===")
	print(f"  Black hole: {config.n_bh:5d} particle   {config.bh_mass:.3e} M☉  (fixed at origin)")
	print(f"  Disk      : {config.n_disk:5d} particles  {config.disk_mass:.3e} M\u2609"
	      f"  (r_d = {config.disk_scale_radius:.1f} kpc)")
	print(f"  Bulge     : {config.n_bulge:5d} particles  {config.bulge_mass:.3e} M\u2609"
	      f"  (a  = {config.bulge_scale_radius:.1f} kpc)")
	print(f"  St. halo  : {config.n_halo:5d} particles  {config.halo_mass:.3e} M☉"
	      f"  (r^-{config.stellar_halo_power:.1f}, r = {config.stellar_halo_inner_radius:.0f}–{config.stellar_halo_radius:.0f} kpc)")
	print(f"  Gas halo  : {config.n_gas:5d} particles  {config.gas_mass:.3e} M☉"
	      f"  (r_max = {config.gas_halo_radius:.1f} kpc)")
	print(f"  Neg. mass : {config.n_negative:5d} particles  {abs(config.neg_particle_mass * config.n_negative):.3e} M☉"
	      f"  (ratio = {config.neg_mass_ratio:.1f}x, r = {config.neg_inner_radius:.0f}–{config.neg_sphere_radius:.0f} kpc"
	      f", σ_v = {config.neg_vel_dispersion:.1f} kpc/Gyr)")
	total_n = config.n_bh + config.n_disk + config.n_bulge + config.n_halo + config.n_gas + config.n_negative
	print(f"  Total N   : {total_n}")
