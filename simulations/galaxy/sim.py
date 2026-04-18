"""
Galactic-scale single-galaxy simulation run loop.

Physical units: kpc, M☉, Gyr throughout.

Key observables computed during the run:
  - Rotation curve  : binned mean tangential velocity v_phi(R) for disk particles
  - Cavity radius   : radius within which negative-mass density falls below half
                      its background level (measures cavity formation progress)
  - Energy          : KE, PE, total (sanity check)
"""

from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp

from antiengine import compute_energy
from antiengine.integrator import make_leapfrog_step, make_elastic_boundary_fn, make_pos_boundary_fn, reinject_escaped_particles
from antiengine.units import kpc_per_Gyr_to_kms
from simulations.galaxy.initial_conditions import GalaxyConfig, initialize_galaxy, print_ic_summary


# ─────────────────────────────────────────────────────────────────────────────
# Rotation curve measurement
# ─────────────────────────────────────────────────────────────────────────────

def measure_rotation_curve(positions, velocities, labels, r_bins):
	"""
	Compute the rotation curve for disk+bulge particles (labels 0 and 1).

	The tangential velocity v_phi is the component perpendicular to the radial
	direction in the x-y plane:
	    v_phi = (x * vy - y * vx) / R

	Parameters
	----------
	positions  : (N, 3) array  [kpc]
	velocities : (N, 3) array  [kpc/Gyr]
	labels     : (N,)   int array
	r_bins     : 1-D array of bin edges [kpc]

	Returns
	-------
	r_centres  : bin centres [kpc]
	v_phi_mean : mean tangential speed per bin [km/s]
	v_phi_std  : std per bin [km/s]
	n_per_bin  : particle count per bin
	"""
	# Use disk + bulge particles only (best tracers of the galactic potential)
	mask = (labels == 0) | (labels == 1)
	pos  = positions[mask]
	vel  = velocities[mask]

	R    = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
	# Signed angular momentum per unit mass around z-axis → tangential speed
	v_phi = (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]) / np.maximum(R, 1e-3)
	# Convert to km/s
	v_phi_kms = v_phi * kpc_per_Gyr_to_kms

	r_centres   = 0.5 * (r_bins[:-1] + r_bins[1:])
	v_phi_mean  = np.zeros(len(r_centres))
	v_phi_std   = np.zeros(len(r_centres))
	n_per_bin   = np.zeros(len(r_centres), dtype=int)

	for i, (r_lo, r_hi) in enumerate(zip(r_bins[:-1], r_bins[1:])):
		in_bin = (R >= r_lo) & (R < r_hi)
		if in_bin.sum() > 0:
			v_phi_mean[i] = v_phi_kms[in_bin].mean()
			v_phi_std[i]  = v_phi_kms[in_bin].std()
			n_per_bin[i]  = in_bin.sum()

	return r_centres, v_phi_mean, v_phi_std, n_per_bin


def measure_cavity_radius(positions, labels, n_bins=60):
	"""
	Find the cavity radius as the location of the steepest density gradient.

	The cavity edge is where neg-mass density transitions from (near) zero
	inside the cavity to a peak just outside it.  The sharpest ascending
	density gradient marks that transition robustly, regardless of how large
	the cavity is or where the outer boundary sits.

	Algorithm:
	  1. Compute neg-mass radial density profile ρ(r) = N_shell / V_shell.
	  2. Smooth with a 3-bin rolling average to reduce Poisson noise.
	  3. Return r at the maximum positive dρ/dr.

	When no cavity has formed, the inner edge of the initial exclusion shell
	is the sharpest transition → returns ≈ neg_inner_radius.

	Returns the cavity radius in kpc, or 0 if there are no neg-mass particles.
	"""
	neg_pos = positions[labels == 4]
	if len(neg_pos) == 0:
		return 0.0

	r_neg = np.linalg.norm(neg_pos, axis=1)
	r_max = float(r_neg.max()) * 1.02   # slightly beyond max for clean last bin

	r_bins      = np.linspace(0, r_max, n_bins + 1)
	r_cents     = 0.5 * (r_bins[:-1] + r_bins[1:])
	shell_vols  = (4 * np.pi / 3) * (r_bins[1:] ** 3 - r_bins[:-1] ** 3)

	counts, _   = np.histogram(r_neg, bins=r_bins)
	density     = counts / shell_vols   # particles per kpc³

	# 3-bin rolling average to smooth Poisson noise
	density_smooth = np.convolve(density, np.ones(3) / 3.0, mode='same')

	# Constrain search to the inner half of the distribution (r < median).
	# Particles pile up at the elastic boundary creating a sharper density jump
	# there than at the inner cavity edge.
	gradient = np.gradient(density_smooth, r_cents)
	r_median = float(np.median(r_neg))
	gradient_inner = gradient.copy()
	gradient_inner[r_cents >= r_median] = 0.0

	max_g = float(gradient_inner.max())
	if max_g <= 0:
		return 0.0

	# Find the innermost local gradient peak above 20% of the maximum.
	# The cavity edge is ALWAYS the leftmost density transition (inner boundary of neg mass).
	# The boundary pile-up gradient is always at LARGER r.
	# Taking the FIRST significant local peak rather than argmax is robust even when many
	# particles have piled up near the wall creating a much larger gradient there.
	threshold = 0.2 * max_g
	n_g = len(gradient_inner)
	for i in range(1, n_g - 1):
		if (gradient_inner[i] >= threshold
				and gradient_inner[i] >= gradient_inner[i - 1]
				and gradient_inner[i] >= gradient_inner[i + 1]):
			return float(r_cents[i])
	# Fallback: leftmost bin above threshold (handles monotone gradients without a local peak)
	above = np.where(gradient_inner >= threshold)[0]
	return float(r_cents[above[0]]) if len(above) > 0 else float(r_cents[int(np.argmax(gradient_inner))])


# ─────────────────────────────────────────────────────────────────────────────
# Effective mass and rotation curve helpers
# ─────────────────────────────────────────────────────────────────────────────

def measure_cavity_effective_mass(positions, labels, masses, r_eval, r_boundary, assume_empty=False, neg_bg_density=0.0):
	"""
	Total effective dark mass = neg-mass deficit inside r_eval versus the
	background density.

	The background density should be the PHYSICAL neg-mass density of the
	universe, not derived from the current simulation boundary (which dilutes
	as the boundary expands, causing M_eff to stay artificially constant).

	When neg_bg_density > 0, it is used directly as the physical background.
	When neg_bg_density == 0, the density is derived from the current
	particle distribution as a fallback (only correct when the boundary
	hasn't expanded from its initial size).

	When assume_empty=True (cavity_from_boundary mode), returns the full
	expected mass without subtracting actual particle count. This is correct
	by the shell theorem: particles displaced to the boundary pile-up shell
	exert zero net force on interior particles and should not reduce M_eff.

	Returns M_eff in M☉.
	"""
	neg_mask = labels == 4
	neg_pos  = positions[neg_mask]
	if len(neg_pos) == 0:
		return 0.0

	N_total = len(neg_pos)
	m_abs   = abs(masses[neg_mask][0])

	# Use the physical background density if provided; otherwise derive
	# from the current state (only accurate when boundary hasn't expanded)
	if neg_bg_density > 0:
		rho_bg_mass = neg_bg_density
	else:
		V_boundary = (4 * np.pi / 3) * r_boundary ** 3
		rho_bg_mass = (N_total * m_abs / V_boundary) if V_boundary > 1e-6 else 0.0

	# Expected mass inside r_eval at background density (full sphere)
	M_expected = rho_bg_mass * (4 * np.pi / 3) * (r_eval ** 3)
	if assume_empty:
		return float(M_expected)

	r_neg    = np.linalg.norm(neg_pos, axis=1)
	M_actual = float((r_neg < r_eval).sum()) * m_abs
	return float(max(M_expected - M_actual, 0.0))


def compute_effective_rotation_curve(positions, labels, masses, config, r_bins, cavity_r=0.0, r_boundary=150.0):
	"""
	Compute v_c(r) = sqrt(G·(M_pos_enc(r) + M_cavity_enc(r)) / r).

	M_cavity_enc is the measured neg-mass deficit at each radius:
	  M_cavity(r) = ρ_bg × (4π/3) × r³  −  M_neg_actual(r)

	When cavity_from_boundary is True (assume_empty mode), the cavity interior
	is treated as fully evacuated:
	  M_cavity(r) = ρ_bg × (4π/3) × min(r, cavity_r)³

	Beyond cavity_r the deficit plateaus at M_cavity(cavity_r) because the shell
	theorem guarantees that neg-mass at r > cavity_r exerts zero net force inside.

	v_pos_only omits the cavity contribution entirely.
	"""
	r_cents   = 0.5 * (r_bins[:-1] + r_bins[1:])
	pos_mask  = labels != 4
	r_pos_all = np.linalg.norm(positions[pos_mask], axis=1)
	m_pos_all = masses[pos_mask]

	neg_mask  = labels == 4
	r_neg_all = np.linalg.norm(positions[neg_mask], axis=1)
	N_neg     = int(neg_mask.sum())
	m_abs     = abs(masses[neg_mask][0]) if N_neg > 0 else 0.0

	# Background density: use physical value if configured, otherwise derive
	cfg_bg_density = getattr(config, 'neg_bg_density', 0.0)
	if cfg_bg_density > 0:
		rho_bg_mass = cfg_bg_density
	else:
		V_boundary = (4 * np.pi / 3) * r_boundary ** 3
		rho_bg_mass = (N_neg * m_abs / V_boundary) if N_neg > 0 and V_boundary > 1e-6 else 0.0
	assume_empty = getattr(config, 'cavity_from_boundary', False)

	# Pre-sort neg radii for efficient cumulative counting
	r_neg_sorted = np.sort(r_neg_all) if N_neg > 0 else np.array([])

	# Total cavity mass at cavity_r (for capping beyond the cavity edge)
	if rho_bg_mass > 0 and cavity_r > 0:
		M_exp_cav = rho_bg_mass * (4 * np.pi / 3) * (cavity_r ** 3)
		if assume_empty:
			M_cavity_at_edge = M_exp_cav
		else:
			N_inside_cav = int(np.searchsorted(r_neg_sorted, cavity_r, side='left'))
			M_act_cav = N_inside_cav * m_abs
			M_cavity_at_edge = max(M_exp_cav - M_act_cav, 0.0)
	else:
		M_cavity_at_edge = 0.0

	v_eff      = np.zeros(len(r_cents))
	v_pos_only = np.zeros(len(r_cents))
	for k, r in enumerate(r_cents):
		M_pos = float(m_pos_all[r_pos_all < r].sum())

		# Cavity dark mass from measured deficit
		if rho_bg_mass > 0:
			if assume_empty:
				# Fully evacuated interior up to cavity_r, plateau beyond
				r_eff = min(r, cavity_r) if cavity_r > 0 else r
				M_cavity = rho_bg_mass * (4 * np.pi / 3) * (r_eff ** 3)
			elif cavity_r > 0 and r > cavity_r:
				# Beyond cavity edge: deficit plateaus
				M_cavity = M_cavity_at_edge
			else:
				# Direct measurement: expected − actual
				M_expected = rho_bg_mass * (4 * np.pi / 3) * (r ** 3)
				N_inside = int(np.searchsorted(r_neg_sorted, r, side='left'))
				M_actual = N_inside * m_abs
				M_cavity = max(M_expected - M_actual, 0.0)
				M_cavity = max(M_expected - M_actual, 0.0)
		else:
			M_cavity = 0.0

		M_total = M_pos + M_cavity
		if M_pos > 0 and r > 0:
			v_pos_only[k] = np.sqrt(config.G * M_pos   / r) * kpc_per_Gyr_to_kms
		if M_total > 0 and r > 0:
			v_eff[k]      = np.sqrt(config.G * M_total / r) * kpc_per_Gyr_to_kms

	return r_cents, v_eff, v_pos_only


# ─────────────────────────────────────────────────────────────────────────────
# Main run loop
# ─────────────────────────────────────────────────────────────────────────────

def run_galaxy_simulation(
	config: GalaxyConfig,
	n_steps:      int = 500,
	record_every: int = 10,
	r_bins=        None,
):
	"""
	Run the single-galaxy simulation and return a history dict.

	Parameters
	----------
	config        : GalaxyConfig
	n_steps       : total number of leapfrog steps
	record_every  : record a snapshot every this many steps
	r_bins        : bin edges for rotation curve [kpc]; default = 0–30 kpc in 20 bins

	Returns
	-------
	history dict with keys:
	  'positions'      : list of (N, 3) numpy arrays  [kpc]
	  'velocities'     : list of (N, 3) numpy arrays  [kpc/Gyr]
	  'labels'         : (N,) int array  (fixed)
	  'masses'         : (N,) float64 array  (fixed, signed)
	  'time'           : list of float  [Gyr]
	  'KE','PE','E'    : list of float  [M☉ kpc² Gyr⁻²]
	  'rotation_curves': list of dicts  {r_cents, v_mean, v_std, n}
	  'cavity_radius'  : list of float  [kpc]
	  'config'         : GalaxyConfig
	"""
	if r_bins is None:
		# 0–80 kpc in 40 bins (2 kpc each).
		# Extending to 80 kpc captures the cavity dark-mass uplift at large
		# radii (neg_inner_radius=20 kpc → cavity effect is only significant
		# at r > ~30 kpc) while the particle rotation curve naturally
		# terminates at the disk truncation radius (~24 kpc).
		r_bins = np.linspace(0, 80.0, 41)

	print_ic_summary(config)
	positions, velocities, masses, labels = initialize_galaxy(config)

	# Transfer to JAX
	pos_j  = jnp.array(positions)
	vel_j  = jnp.array(velocities)
	mass_j = jnp.array(masses)

	step_fn = make_leapfrog_step(mass_j, config.G, config.softening, config.dt,
							   interaction_mode=config.interaction_mode)

	history = {
		'positions':       [],
		'velocities':      [],
		'labels':          labels,
		'masses':          masses,
		'time':            [],
		'KE':              [],
		'PE':              [],
		'E':               [],
		'rotation_curves':           [],
		'effective_rotation_curves': [],   # analytical v_c including cavity dark-mass contribution
		'pos_only_rotation_curves':  [],   # analytical v_c from enclosed positive mass only (no cavity)
		'cavity_radius':             [],
		'effective_mass':            [],   # effective positive mass from cavity [M☉]
		'boundary_radius':           [],   # neg-mass boundary radius over time [kpc]
		'config':                    config,
	}

	print(f"\nRunning {n_steps} steps × dt={config.dt} Gyr "
	      f"= {n_steps * config.dt:.2f} Gyr  (recording every {record_every})")
	print(f"N_total={len(masses)}  softening={config.softening} kpc"
	      f"  interaction_mode={config.interaction_mode}")
	if config.neg_bg_density > 0:
		print(f"neg_bg_density={config.neg_bg_density:.1f} M☉/kpc³ → neg_sphere_radius={config.neg_sphere_radius:.1f} kpc")
	print()

	# Warm up JIT
	_ = step_fn(pos_j, vel_j)
	# Boundary setup: choose between elastic reflection and re-injection.
	# - Elastic: reflects off bubble wall (models neighbouring cavities pressing back)
	# - Re-injection: teleports to boundary surface with inward velocity
	#   (models infinite neg-mass background — our sim is a window into the universe)
	boundary_fn = None
	use_reinjection = config.use_reinjection_boundary
	r_boundary  = config.neg_sphere_radius
	rng = np.random.default_rng(config.seed + 1)  # separate stream from IC sampling
	# Positive-mass boundary: confine gas/halo particles within the simulation domain
	pos_boundary_fn = make_pos_boundary_fn(mass_j)
	_ = pos_boundary_fn(pos_j, vel_j, jnp.array(r_boundary))  # warm up JIT
	if use_reinjection:
		print(f"Using re-injection boundary (vel_dispersion={config.neg_vel_dispersion:.1f} kpc/Gyr)")
	elif config.use_elastic_boundary:
		boundary_fn = make_elastic_boundary_fn(mass_j, config.boundary_restitution)
		_ = boundary_fn(pos_j, vel_j, jnp.array(r_boundary))  # warm up boundary JIT
	if config.hubble_expansion_boundary:
		print(f"Hubble expansion boundary: H₀={config.hubble_rate:.5f} Gyr⁻¹")
	print("JIT compilation complete — starting simulation...")

	neg_offset = config.n_bh + config.n_disk + config.n_bulge + config.n_halo + config.n_gas
	neg_np_mask = labels == 4   # boolean mask for neg particles in numpy arrays
	# Peak cavity radius: physical cavity only grows; transient dips from a few
	# reflected particles bouncing back in are measurement noise, not real collapse.
	peak_cavity_r = 0.0

	t = 0.0
	for i in range(n_steps):
		pos_j, vel_j = step_fn(pos_j, vel_j)
		# Pin the central BH to the origin — it is a gravitational anchor, not an orbiter
		if config.n_bh > 0:
			pos_j = pos_j.at[:config.n_bh].set(jnp.zeros((config.n_bh, 3)))
			vel_j = vel_j.at[:config.n_bh].set(jnp.zeros((config.n_bh, 3)))
		if boundary_fn is not None:
			vel_j = boundary_fn(pos_j, vel_j, jnp.array(r_boundary))
		# Re-injection boundary: teleport escaped neg-mass to boundary surface
		if use_reinjection:
			pos_np = np.array(pos_j)
			vel_np = np.array(vel_j)
			reinject_escaped_particles(
				pos_np, vel_np, neg_offset, r_boundary,
				config.neg_vel_dispersion, rng,
			)
			pos_j = jnp.array(pos_np)
			vel_j = jnp.array(vel_np)
		# Confine positive-mass particles (gas, halo) within the boundary
		vel_j = pos_boundary_fn(pos_j, vel_j, jnp.array(r_boundary))
		# Grow bubble: fixed rate + coupling to mean outward velocity of near-wall particles
		if config.use_elastic_boundary and config.boundary_expansion_coupling > 0:
			neg_p  = np.array(pos_j[neg_offset:])
			neg_v  = np.array(vel_j[neg_offset:])
			r_neg  = np.linalg.norm(neg_p, axis=1)
			r_hat  = neg_p / np.maximum(r_neg, 1e-10)[:, None]
			v_rad  = (neg_v * r_hat).sum(axis=1)
			near   = r_neg > 0.85 * r_boundary
			mean_v_out = float(max(0.0, v_rad[near].mean())) if near.any() else 0.0
			r_boundary += config.boundary_expansion_coupling * mean_v_out * config.dt
		r_boundary += config.boundary_expansion_rate * config.dt
		# Hubble expansion of the boundary
		if config.hubble_expansion_boundary:
			r_boundary *= (1.0 + config.hubble_rate * config.dt)
		t += config.dt

		if i % record_every == 0:
			pos_np = np.array(pos_j)
			vel_np = np.array(vel_j)

			KE, PE, E = compute_energy(pos_j, vel_j, mass_j, config.G, config.softening,
									   interaction_mode=config.interaction_mode)
			r_cents, v_mean, v_std, n_bin = measure_rotation_curve(
				pos_np, vel_np, labels, r_bins
			)
			cav_r_raw = measure_cavity_radius(pos_np, labels)
			peak_cavity_r = max(peak_cavity_r, cav_r_raw)
			cav_r = cav_r_raw   # display raw value; M_eff uses peak for stability
			# Cavity radius for M_eff: either measured or boundary
			eff_cav_r = r_boundary if config.cavity_from_boundary else peak_cavity_r
			eff_m = measure_cavity_effective_mass(
				pos_np, labels, masses,
				r_eval=eff_cav_r,
				r_boundary=r_boundary,
				assume_empty=config.cavity_from_boundary,
				neg_bg_density=config.neg_bg_density,
			)
			r_eff_cents, v_eff, v_pos_only = compute_effective_rotation_curve(
				pos_np, labels, masses, config, r_bins,
				cavity_r=eff_cav_r,
				r_boundary=r_boundary,
			)

			history['positions'].append(pos_np)
			history['velocities'].append(vel_np)
			history['time'].append(t)
			history['KE'].append(float(KE))
			history['PE'].append(float(PE))
			history['E'].append(float(E))
			history['rotation_curves'].append({
				'r':     r_cents,
				'v':     v_mean,
				'v_std': v_std,
				'n':     n_bin,
			})
			history['effective_rotation_curves'].append({
				'r': r_eff_cents,
				'v': v_eff,
			})
			history['pos_only_rotation_curves'].append({
				'r': r_eff_cents,
				'v': v_pos_only,
			})
			history['cavity_radius'].append(cav_r)
			history['effective_mass'].append(eff_m)
			history['boundary_radius'].append(r_boundary)

			E0    = history['E'][0]
			fluct = abs(float(E) - E0) / abs(E0) * 100 if E0 != 0 else 0.0
			bnd_info = f"  r_bnd={r_boundary:.1f}" if config.hubble_expansion_boundary else ""
			print(f"  t={t:.3f} Gyr  |ΔE/E₀|={fluct:.3f}%{bnd_info}")

	n_frames = len(history['positions'])
	print(f"\nDone. {n_frames} frames recorded over {t:.3f} Gyr.")
	return history


def print_galaxy_diagnostics(history):
	"""Print a summary of energy conservation and cavity growth."""
	E = np.array(history['E'])
	E0 = E[0]
	max_fluct = np.max(np.abs(E - E0)) / abs(E0) * 100 if E0 != 0 else float('nan')
	rms_fluct = np.std(E) / abs(E0) * 100 if E0 != 0 else float('nan')
	drift     = (E[-1] - E0) / abs(E0) * 100 if E0 != 0 else float('nan')
	times     = np.array(history['time'])
	cavities  = np.array(history['cavity_radius'])

	print("\n=== Galaxy simulation diagnostics ===")
	print(f"  Duration         : {times[-1]:.3f} Gyr")
	print(f"  Energy drift     : {drift:.4f}%")
	print(f"  Energy RMS fluct : {rms_fluct:.4f}%")
	print(f"  Energy max fluct : {max_fluct:.4f}%")

	print(f"\n  Cavity radius at start : {cavities[0]:.1f} kpc")
	print(f"  Cavity radius at end   : {cavities[-1]:.1f} kpc")

	# Rotation curve at first and last snapshot
	rc_first = history['rotation_curves'][0]
	rc_last  = history['rotation_curves'][-1]
	print(f"\n  Rotation curve (selected bins):")
	print(f"    {'r [kpc]':>10}  {'v_t=0 [km/s]':>14}  {'v_final [km/s]':>16}")
	for j in range(0, len(rc_first['r']), 4):
		r  = rc_first['r'][j]
		v0 = rc_first['v'][j]
		vf = rc_last['v'][j]
		if rc_first['n'][j] > 0:
			print(f"    {r:10.1f}  {v0:14.1f}  {vf:16.1f}")
