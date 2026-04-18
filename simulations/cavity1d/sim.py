"""
1D Spherical Cavity Simulation

Tracks N radial Lagrangian shells of negative mass in spherical symmetry around
a central galaxy (fixed point mass M_gal). Each shell has radius r_i and radial
velocity v_i, evolved with leapfrog (KDK) integration.

Physics:
  The force on a neg-mass shell at radius r has two components:
    1. Galaxy repulsion (pos-neg): pushes shell outward  (+G M_gal / r²)
    2. Neg-mass perturbation: the deviation from a uniform background density

  In an infinite uniform neg-mass background, the net self-force is zero by
  symmetry (Jeans swindle). Only perturbations from uniformity contribute.
  The perturbation δM_enc(r) = M_neg_enc(r) - (4π/3)ρ_bg r³ gives:

    a(r) = G × [M_gal + δM_enc(r)] / r²

  Inside a cavity (less neg mass than background), δM < 0 → restoring force
  that opposes the galaxy repulsion. Equilibrium when M_gal = (4π/3)ρ_bg R_eq³.

  This is the self-limiting mechanism: the cavity cannot evacuate more mass than
  M_gal. Beyond R_eq, the background restoring force exceeds the galaxy push.

  In the finite sphere model (no Jeans swindle), both galaxy and interior neg-neg
  repulsion push outward with no restoring force — all shells hit the boundary.
  The Jeans swindle version (use_jeans_swindle=True) is the physically correct
  model for a galaxy embedded in an infinite neg-mass background.

Boundary conditions:
  Shells beyond R_domain are re-injected at the boundary with configurable inward
  velocity, representing an infinite neg-mass reservoir.

Units: kpc, M☉, Gyr (project convention).
"""

import numpy as np
from dataclasses import dataclass
from antiengine.units import G_kpc_Msun_Gyr, kpc_per_Gyr_to_kms


@dataclass
class Cavity1DConfig:
	"""Configuration for the 1D spherical cavity simulation."""
	# Galaxy
	galaxy_mass: float = 1.25e11		# M☉ — MW/Andromeda scale

	# Negative mass background
	rho_bg: float = 3.9e4				# M☉/kpc³ — cosmological neg-mass density
	r_domain: float = 300.0				# kpc — simulation domain radius

	# Shells
	n_shells: int = 2000				# number of Lagrangian neg-mass shells

	# Integration
	dt: float = 0.001					# Gyr (1 Myr timestep)
	n_steps: int = 5000					# total steps (5 Gyr)
	record_every: int = 50				# snapshot interval

	# Boundary
	reinject: bool = True				# reinject shells that leave the domain
	inflow_velocity: float = 0.0		# kpc/Gyr — mean inward velocity at boundary (0 = stationary)
	velocity_dispersion: float = 0.0	# kpc/Gyr — thermal velocity dispersion


def compute_shell_accelerations(radii, shell_masses, galaxy_mass, rho_bg, G):
	"""
	Compute radial acceleration for each neg-mass shell using the correct force law.

	For a neg-mass shell at radius r in a finite domain:
		a(r) = +G × [M_gal + M_neg_enc(r)] / r²

	Both the galaxy (pos-neg repulsion) and the interior neg mass (neg-neg repulsion)
	push the shell outward. M_neg_enc is the magnitude of neg mass inside r.

	To model the restoring effect of the infinite background, we use the Jeans swindle:
	work with perturbations from uniform density. The effective force from the uniform
	background is obtained by subtracting the force that a uniform sphere of neg mass
	would exert:

	With background subtraction (Jeans swindle):
		a(r) = +G M_gal / r²  (galaxy repulsion)
		     + G [M_neg_enc(r) - (4π/3) ρ_bg r³] / r²  (perturbation from neg mass)

	When the cavity is evacuated (M_neg_enc ≈ 0):
		a(r) = +G M_gal / r² - G (4π/3) ρ_bg r³ / r²
		     = G [M_gal - (4π/3) ρ_bg r³] / r²

	This gives zero at R_eq = (3 M_gal / 4π ρ_bg)^(1/3), i.e. the self-limiting radius.

	The Jeans swindle version represents an infinite medium where the uniform background
	is subtracted. The no-swindle version represents a finite isolated sphere.

	We implement BOTH and let the user choose via the boundary conditions:
	- With reinjection (infinite medium model): use Jeans swindle
	- Without reinjection (finite sphere): use raw forces (no background subtraction)
	"""
	sort_idx = np.argsort(radii)
	unsort_idx = np.argsort(sort_idx)

	sorted_radii = radii[sort_idx]
	sorted_masses = shell_masses[sort_idx]

	# Cumulative enclosed neg mass (magnitude)
	cumulative_mass = np.cumsum(np.abs(sorted_masses))

	# Background mass that would fill each shell's radius
	bg_enclosed = (4.0 / 3.0) * np.pi * rho_bg * sorted_radii**3

	# Perturbation: actual neg mass minus background expectation
	delta_M = cumulative_mass - bg_enclosed

	# Force from galaxy (outward) + perturbation from neg-mass rearrangement
	# The perturbation δM acts via neg-neg interaction:
	#   δM > 0 (more neg mass than bg) → extra repulsion → outward
	#   δM < 0 (less neg mass than bg) → deficit → "restoring" → inward
	r_safe = np.maximum(sorted_radii, 1e-6)  # avoid division by zero
	accel = G * (galaxy_mass + delta_M) / r_safe**2

	return accel[unsort_idx]


def compute_shell_accelerations_raw(radii, shell_masses, galaxy_mass, G):
	"""
	Raw (no Jeans swindle) acceleration — finite isolated sphere model.

	a(r) = G × [M_gal + M_neg_enc(r)] / r²

	Everything pushes neg mass outward. No restoring force.
	"""
	sort_idx = np.argsort(radii)
	unsort_idx = np.argsort(sort_idx)

	sorted_radii = radii[sort_idx]
	sorted_masses = shell_masses[sort_idx]

	cumulative_mass = np.cumsum(np.abs(sorted_masses))

	r_safe = np.maximum(sorted_radii, 1e-6)
	accel = G * (galaxy_mass + cumulative_mass) / r_safe**2

	return accel[unsort_idx]


def run_cavity_1d(config: Cavity1DConfig, use_jeans_swindle=True):
	"""
	Run the 1D spherical cavity simulation.

	Parameters
	----------
	config : Cavity1DConfig
	use_jeans_swindle : bool
		If True, use Jeans swindle (background subtraction) to model infinite medium.
		If False, use raw forces (finite sphere, no restoring force).

	Returns
	-------
	history : dict with 'times', 'radii', 'velocities', 'density_profiles', 'config'
	"""
	G = G_kpc_Msun_Gyr
	rng = np.random.default_rng(42)

	# Initialize shells uniformly distributed in volume (r ∝ u^(1/3))
	u = np.linspace(0.01, 1.0, config.n_shells)
	radii = config.r_domain * u**(1.0 / 3.0)
	velocities = np.zeros(config.n_shells)

	# Each shell represents an equal fraction of the total neg mass in the domain
	total_neg_mass = (4.0 / 3.0) * np.pi * config.rho_bg * config.r_domain**3
	shell_mass = total_neg_mass / config.n_shells  # magnitude per shell

	# Softening to prevent shells at r→0 from diverging
	r_soft = 0.5  # kpc

	# Equilibrium radius for reference
	R_eq = (3.0 * config.galaxy_mass / (4.0 * np.pi * config.rho_bg))**(1.0 / 3.0)

	print(f"=== 1D Spherical Cavity Simulation ===")
	print(f"  Galaxy mass:       {config.galaxy_mass:.2e} M☉")
	print(f"  Background ρ:      {config.rho_bg:.2e} M☉/kpc³")
	print(f"  Domain radius:     {config.r_domain:.0f} kpc")
	print(f"  Equilibrium R_eq:  {R_eq:.1f} kpc")
	print(f"  Total neg mass:    {total_neg_mass:.2e} M☉")
	print(f"  Shell mass:        {shell_mass:.2e} M☉")
	print(f"  N shells:          {config.n_shells}")
	print(f"  dt:                {config.dt} Gyr ({config.dt*1e3:.1f} Myr)")
	print(f"  Total time:        {config.n_steps * config.dt:.2f} Gyr")
	print(f"  Jeans swindle:     {use_jeans_swindle}")
	print()

	# History storage
	times = []
	radii_history = []
	vel_history = []

	shell_masses = np.full(config.n_shells, shell_mass)

	for step in range(config.n_steps):
		# Record snapshot
		if step % config.record_every == 0:
			t = step * config.dt
			times.append(t)
			radii_history.append(radii.copy())
			vel_history.append(velocities.copy())

			if step % (config.record_every * 10) == 0:
				n_inside = np.sum(radii < R_eq)
				n_outside = np.sum(radii >= R_eq)
				print(f"  t={t:.3f} Gyr | inside R_eq: {n_inside} | "
					  f"outside: {n_outside} | "
					  f"r_min={radii.min():.1f} r_max={radii.max():.1f} kpc")

		# Compute accelerations (leapfrog KDK)
		if use_jeans_swindle:
			accel = compute_shell_accelerations(
				radii, shell_masses, config.galaxy_mass, config.rho_bg, G)
		else:
			accel = compute_shell_accelerations_raw(
				radii, shell_masses, config.galaxy_mass, G)

		# Apply softening at small radii: replace r² with r²+ε²
		# Already handled in the acceleration functions via r_safe, but add
		# an explicit softening to prevent shells from overshooting the center
		r_safe = np.maximum(radii, r_soft)
		softening_factor = (radii / r_safe)**2
		accel *= softening_factor

		# Half-kick
		velocities += 0.5 * config.dt * accel

		# Drift
		radii += config.dt * velocities

		# Handle shells that cross r=0 (reflect)
		crossed = radii < 0
		radii[crossed] = -radii[crossed]
		velocities[crossed] = -velocities[crossed]

		# Recompute acceleration at new position
		if use_jeans_swindle:
			accel = compute_shell_accelerations(
				radii, shell_masses, config.galaxy_mass, config.rho_bg, G)
		else:
			accel = compute_shell_accelerations_raw(
				radii, shell_masses, config.galaxy_mass, G)

		r_safe = np.maximum(radii, r_soft)
		softening_factor = (radii / r_safe)**2
		accel *= softening_factor

		# Half-kick
		velocities += 0.5 * config.dt * accel

		# Boundary: reinject or reflect
		if config.reinject:
			escaped = radii > config.r_domain
			n_escaped = np.sum(escaped)
			if n_escaped > 0:
				# Reinject at domain boundary with inward velocity
				radii[escaped] = config.r_domain - rng.uniform(0, 1, n_escaped) * 0.1
				if config.velocity_dispersion > 0:
					velocities[escaped] = -np.abs(
						rng.normal(config.inflow_velocity, config.velocity_dispersion, n_escaped))
				else:
					velocities[escaped] = -config.inflow_velocity
		else:
			# Reflect off boundary
			escaped = radii > config.r_domain
			radii[escaped] = 2 * config.r_domain - radii[escaped]
			velocities[escaped] = -np.abs(velocities[escaped])

	# Final snapshot
	times.append(config.n_steps * config.dt)
	radii_history.append(radii.copy())
	vel_history.append(velocities.copy())

	print(f"\n  Simulation complete: t={config.n_steps * config.dt:.2f} Gyr")

	return {
		'times': np.array(times),
		'radii': np.array(radii_history),
		'velocities': np.array(vel_history),
		'config': config,
		'R_eq': R_eq,
		'shell_mass': shell_mass,
		'use_jeans_swindle': use_jeans_swindle,
	}


def measure_density_profile(radii_snapshot, shell_mass, r_bins):
	"""
	Measure the neg-mass density profile from a shell radii snapshot.

	Returns bin centres and density in each radial bin (M☉/kpc³).
	"""
	r_centres = 0.5 * (r_bins[:-1] + r_bins[1:])
	vol = (4.0 / 3.0) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
	counts, _ = np.histogram(radii_snapshot, bins=r_bins)
	density = counts * shell_mass / vol
	return r_centres, density


def measure_enclosed_mass(radii_snapshot, shell_mass, r_eval):
	"""
	Compute the enclosed neg-mass magnitude at each evaluation radius.
	"""
	r_eval = np.asarray(r_eval)
	M_enc = np.array([np.sum(radii_snapshot <= r) * shell_mass for r in r_eval])
	return M_enc


def compute_rotation_curve_contribution(radii_snapshot, shell_mass, rho_bg, r_eval):
	"""
	Compute the cavity's contribution to the rotation curve.

	The effective DM mass (deficit) at radius r:
		M_DM(r) = (4π/3) ρ_bg r³ - M_neg_enc(r)

	The circular velocity contribution:
		v_c(r) = sqrt(G × M_DM(r) / r)  if M_DM > 0
	"""
	G = G_kpc_Msun_Gyr
	r_eval = np.asarray(r_eval, dtype=float)
	M_neg_enc = measure_enclosed_mass(radii_snapshot, shell_mass, r_eval)
	M_bg = (4.0 / 3.0) * np.pi * rho_bg * r_eval**3
	M_DM = M_bg - M_neg_enc
	M_DM = np.maximum(M_DM, 0)  # can't be negative (over-evacuated)
	v_c = np.sqrt(G * M_DM / np.maximum(r_eval, 0.1))
	v_c_kms = v_c * kpc_per_Gyr_to_kms
	return r_eval, v_c_kms, M_DM
