"""
Phase 3: Cosmological simulation run loop.

Evolves particles in comoving coordinates with periodic boundary conditions,
PM gravity, and a self-consistent scale factor from the Friedmann acceleration
equation.

Comoving equations of motion (super-comoving time τ where dτ = dt/a²):
  dx/dτ = p                  (comoving momentum)
  dp/dτ = a × F_pert         (perturbation force × scale factor)
  da/dτ = a² ȧ = a² × a × H
  d(aH)/dτ = a² × ä         (from Friedmann acceleration eq.)

Using cosmic time t directly with KDK leapfrog:
  p = a² ẋ  (canonical momentum = a² × peculiar velocity)

  Half-kick:  p += ½ dt × F_pert / a
  Drift:      x += dt × p / a²
  Half-kick:  p += ½ dt × F_pert(x_new) / a

  The 1/a factor arises because the PM solver computes φ_code satisfying
  ∇²_x φ = 4πG δρ_comov (no factor of a), giving φ_code = a × δΦ_phys.
  So dp/dt = -∇_x δΦ_phys = -(1/a) ∇_x φ_code = accels / a.

  Scale factor: ä = (4πG/3)(|ρ̄_neg| - ρ̄_pos) / a²

For 50/50 ratio: ä = 0 → constant velocity expansion (Milne/coasting universe).
The PM solver handles perturbation forces; the mean density drives expansion.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from antiengine.units import G_kpc_Msun_Gyr
from .gravity import pm_accelerations


# ─────────────────────────────────────────────────────────────────────────────
# Effective Hubble rate measurement (backreaction diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def _measure_effective_hubble(positions, momenta, masses, a, adot, box_size,
							 n_sample=200, rng=None):
	"""
	Measure the effective Hubble rate from particle pair recession velocities.

	For a pair (i, j), the total physical recession velocity is:
	  v_rec = ȧ Δx + Δp / a
	and the physical separation is:
	  d = a |Δx|

	Fitting v_rec = H_eff × d via linear regression through the origin gives:
	  H_eff = ȧ/a + Σ(Δp·Δx) / (a² Σ|Δx|²)

	The backreaction correction δH = H_eff - ȧ/a measures how species
	segregation has modified the expansion rate relative to the homogeneous
	Friedmann equation. Positive δH means expansion is faster than Milne.

	Returns (H_eff_pos, H_eff_neg).
	"""
	positions_np = np.asarray(positions)
	momenta_np = np.asarray(momenta)
	masses_np = np.asarray(masses)
	L = box_size
	H_bg = adot / a if a > 1e-10 else 0.0

	H_pos, H_neg = H_bg, H_bg

	for species in ('pos', 'neg'):
		mask = masses_np > 0 if species == 'pos' else masses_np < 0
		idx = np.where(mask)[0]
		if len(idx) < 2:
			continue

		# Subsample for performance
		n = min(n_sample, len(idx))
		chosen = rng.choice(idx, n, replace=False) if len(idx) > n_sample else idx

		x = positions_np[chosen]  # (n, 3)
		p = momenta_np[chosen]    # (n, 3)

		# All-pairs differences
		dx = x[:, None, :] - x[None, :, :]  # (n, n, 3)
		dp = p[:, None, :] - p[None, :, :]  # (n, n, 3)

		# Minimum image convention (nearest periodic image)
		dx = dx - L * np.round(dx / L)

		# Upper triangle only (avoid self-pairs and double-counting)
		tri_i, tri_j = np.triu_indices(n, k=1)
		dx_pairs = dx[tri_i, tri_j]  # (n_pairs, 3)
		dp_pairs = dp[tri_i, tri_j]  # (n_pairs, 3)

		dx_sq = np.sum(dx_pairs**2, axis=1)
		dp_dot_dx = np.sum(dp_pairs * dx_pairs, axis=1)

		sum_dx_sq = np.sum(dx_sq)
		sum_dp_dx = np.sum(dp_dot_dx)

		if sum_dx_sq > 0:
			H_eff = H_bg + sum_dp_dx / (a**2 * sum_dx_sq)
		else:
			H_eff = H_bg

		if species == 'pos':
			H_pos = H_eff
		else:
			H_neg = H_eff

	return H_pos, H_neg


# ─────────────────────────────────────────────────────────────────────────────
# Scale factor evolution (Friedmann)
# ─────────────────────────────────────────────────────────────────────────────

def _friedmann_accel(a, rho_pos_0, rho_neg_0, G_const):
	"""
	Second Friedmann equation for the anti-universe model.

	ä/a = (4πG/3) × [|ρ̄_neg,0| - ρ̄_pos,0] / a³

	Positive mass decelerates expansion (gravity).
	Negative mass accelerates expansion (anti-gravity).
	Both dilute as 1/a³ (matter).

	Returns ä (acceleration of the scale factor).
	"""
	rho_eff = (rho_neg_0 - rho_pos_0) / a**3
	return a * (4.0 * np.pi * G_const / 3.0) * rho_eff


# ─────────────────────────────────────────────────────────────────────────────
# Power spectrum measurement
# ─────────────────────────────────────────────────────────────────────────────

def measure_power_spectrum(positions, masses, box_size, n_grid, mask=None):
	"""
	Measure the 1D spherically-averaged power spectrum P(k).

	positions : (N, 3) comoving positions
	masses    : (N,)   particle masses (absolute values used)
	box_size  : box side length
	n_grid    : grid cells per dimension
	mask      : optional boolean mask to select a subset of particles

	Returns (k_bins, P_k) — wavenumber and power spectrum arrays.
	"""
	if mask is not None:
		positions = positions[mask]
		masses = masses[mask]

	from .gravity import cic_assign
	abs_masses = np.abs(masses)

	rho = np.array(cic_assign(
		jnp.array(positions), jnp.array(abs_masses), box_size, n_grid
	))
	rho_mean = np.mean(rho)
	if rho_mean > 0:
		delta = (rho - rho_mean) / rho_mean
	else:
		delta = rho * 0.0

	delta_k = np.fft.fftn(delta)
	pk_3d = np.abs(delta_k) ** 2 / n_grid**3

	# Spherical average
	freqs = np.fft.fftfreq(n_grid, d=box_size / n_grid)
	kx, ky, kz = np.meshgrid(freqs, freqs, freqs, indexing='ij')
	k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

	k_fund = 1.0 / box_size
	k_nyq = n_grid / (2.0 * box_size)
	n_bins = n_grid // 2
	k_edges = np.linspace(k_fund, k_nyq, n_bins + 1)
	k_bins = 0.5 * (k_edges[:-1] + k_edges[1:])

	pk_1d = np.zeros(n_bins)
	for i in range(n_bins):
		shell = (k_mag >= k_edges[i]) & (k_mag < k_edges[i + 1])
		if np.any(shell):
			pk_1d[i] = np.mean(pk_3d[shell])

	return k_bins, pk_1d


# ─────────────────────────────────────────────────────────────────────────────
# Separation metric
# ─────────────────────────────────────────────────────────────────────────────

def measure_separation(positions, masses, box_size, n_grid):
	"""
	Measure the cross-correlation between positive and negative density fields.

	Returns the Pearson correlation coefficient r ∈ [-1, +1].
	  r ≈ +1 : species overlap (no separation)
	  r ≈  0 : uncorrelated
	  r ≈ -1 : anti-correlated (complete separation — pos in voids of neg)
	"""
	from .gravity import cic_assign

	pos_mask = masses > 0
	neg_mask = masses < 0
	abs_m = np.abs(masses)

	rho_pos = np.array(cic_assign(
		jnp.array(positions), jnp.array(abs_m * pos_mask), box_size, n_grid,
	))
	rho_neg = np.array(cic_assign(
		jnp.array(positions), jnp.array(abs_m * neg_mask), box_size, n_grid,
	))

	# Pearson correlation of the two density fields
	dp = rho_pos - np.mean(rho_pos)
	dn = rho_neg - np.mean(rho_neg)
	numerator = np.sum(dp * dn)
	denominator = np.sqrt(np.sum(dp**2) * np.sum(dn**2))
	if denominator > 0:
		return numerator / denominator
	return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main run loop
# ─────────────────────────────────────────────────────────────────────────────

def run_cosmological_simulation(config, n_steps, record_every=10):
	"""
	Run the cosmological simulation.

	Uses leapfrog (KDK) in comoving coordinates with:
	  - PM gravity for perturbation forces
	  - Self-consistent Friedmann scale factor (or backreaction-corrected)
	  - Periodic boundary conditions

	When config.backreaction is True, the scale factor is driven by the
	effective Hubble rate measured from positive-mass particle dynamics
	instead of the homogeneous Friedmann equation. This captures the
	shell-theorem expansion effect from species segregation.

	The effective scale factor a_eff(t) and effective Hubble rates are
	always recorded as diagnostics, regardless of the backreaction flag.

	Returns a history dict with recorded snapshots and diagnostics.
	"""
	from .initial_conditions import initialize_cosmological

	# ── Generate ICs ─────────────────────────────────────────────────────────
	positions, velocities, masses, labels = initialize_cosmological(config)

	# Convert to JAX arrays
	positions = jnp.array(positions)
	velocities = jnp.array(velocities)
	masses = jnp.array(masses)
	labels_np = np.array(labels)

	n_total = positions.shape[0]
	n_pos = config.n_particles
	n_neg = config.n_particles
	L = config.box_size

	# ── Scale factor initial conditions ──────────────────────────────────────
	a = config.a_initial

	# Mean comoving densities (constant — comoving volume is fixed)
	rho_pos_0 = config.pos_total_mass / L**3
	rho_neg_0 = config.neg_total_mass / L**3

	# First Friedmann equation: H²(a) = H₀² × [Ω_eff/a³ + Ω_k/a²]
	# where Ω_eff = (ρ_pos - ρ_neg) / ρ_crit and Ω_k = 1 - Ω_eff.
	# For 50/50: Ω_eff=0, Ω_k=1 → H(a)=H₀/a → ȧ=H₀ (Milne coasting)
	rho_crit = 3.0 * config.H0**2 / (8.0 * np.pi * config.G)
	omega_eff = (rho_pos_0 - rho_neg_0) / rho_crit
	omega_k = 1.0 - omega_eff
	H2_initial = config.H0**2 * (omega_eff / a**3 + omega_k / a**2)
	if H2_initial < 0:
		raise ValueError(
			f"H²(a₀) < 0: scale factor {a} is below the bounce scale for "
			f"this mass ratio. Increase a_initial."
		)
	H_initial = np.sqrt(H2_initial)
	adot = a * H_initial  # ȧ = a × H(a₀)

	# ── Timestep ─────────────────────────────────────────────────────────────
	# Estimate total time from a_initial to a_final under Milne (ȧ=const):
	# For R≠1 (accelerating models near bounce), the Milne estimate can be
	# wildly wrong because ȧ₀ is tiny near the bounce. Use numerical
	# integration of the Friedmann equation for an accurate t_total.
	a_final = config.a_final
	if omega_eff < 0:
		from scipy.integrate import quad as _quad_fn
		def _t_integrand(a_val):
			H2 = config.H0**2 * (omega_eff / a_val**3 + omega_k / a_val**2)
			if H2 <= 0:
				return 0.0
			return 1.0 / (a_val * np.sqrt(H2))
		t_total, _ = _quad_fn(_t_integrand, a, a_final, limit=200)
	else:
		t_total = (a_final - a) / adot if adot > 0 else 1.0 / config.H0

	# ── History storage ──────────────────────────────────────────────────────
	history = {
		'positions': [],
		'labels': labels_np,
		'masses': np.array(masses),
		'box_size': L,
		'n_grid': config.n_grid,
		'config': config,
		# Time series
		'time': [],
		'scale_factor': [],
		'hubble_rate': [],
		# Backreaction diagnostics
		'effective_hubble_pos': [],
		'effective_hubble_neg': [],
		'effective_scale_factor': [],
		# Power spectra (recorded less frequently)
		'pk_pos': [],
		'pk_neg': [],
		'k_bins': None,
		# Separation metric
		'cross_correlation': [],
	}

	# ── Initial PM acceleration ──────────────────────────────────────────────
	print(f"\nStarting cosmological simulation: {n_steps} steps")
	print(f"  n_total = {n_total}, n_grid = {config.n_grid}³")
	print(f"  a_initial = {a:.4f}, H(a₀) = {H_initial:.5f} Gyr⁻¹, ȧ₀ = {adot:.5f}")
	print(f"  Ω_eff = {omega_eff:.4f}, Ω_k = {omega_k:.4f}")
	print(f"  ρ̄_pos = {rho_pos_0:.2f} M☉/kpc³, ρ̄_neg = {rho_neg_0:.2f} M☉/kpc³")
	print(f"  t_total ≈ {t_total:.1f} Gyr, dt ≈ {t_total / n_steps:.4f} Gyr")

	# Canonical momentum p = a² × ẋ (peculiar velocity in comoving coords)
	momenta = a**2 * velocities

	# Compute initial accelerations
	t_start = time.time()
	accels = pm_accelerations(positions, masses, config.G, L, config.n_grid)
	jax.block_until_ready(accels)
	t_first = time.time() - t_start
	print(f"  First PM solve: {t_first:.2f}s (includes JIT compilation)")

	# ── Backreaction state ───────────────────────────────────────────────────
	backreaction_rng = np.random.default_rng(config.seed + 999)
	a_eff = a                     # effective scale factor (from measured H_eff)
	delta_H_smooth = 0.0          # EMA-smoothed backreaction correction (H_eff - H_bg)
	adot_friedmann = adot         # Friedmann-only ȧ (no backreaction corrections)
	if config.backreaction:
		print(f"  Backreaction mode: ON (n_sample={config.backreaction_n_sample})")

	# Most recent H_eff measurements (used by backreaction mode)
	last_H_eff_pos = H_initial
	last_H_eff_neg = H_initial

	# ── Time integration (KDK leapfrog) ─────────────────────────────────────
	t_sim = 0.0

	for step in range(n_steps):
		# Adaptive timestep: base from total time, clamped to Hubble fraction
		H = adot / a if a > 0 else H_initial
		dt = min(t_total / n_steps, 0.05 / max(abs(H), 1e-10))

		# ── Half-kick: p += ½ dt × F_pert / a ──────────────────────────
		momenta = momenta + 0.5 * dt * accels / a

		# ── Drift: x += dt × p / a² ──────────────────────────────────────
		positions = positions + dt * momenta / a**2

		# ── Evolve scale factor ───────────────────────────────────────────
		a_old = a  # save for a_eff integration

		# Friedmann acceleration (used in both branches)
		a_accel = _friedmann_accel(a, rho_pos_0, rho_neg_0, config.G)

		if config.backreaction:
			# Friedmann Verlet as the base, with δH correction applied
			# non-cumulatively to the Hubble rate.
			#
			# The δH correction modifies the instantaneous Hubble rate, not
			# the acceleration. We track adot_friedmann separately so that
			# corrections don't accumulate in the velocity state.
			H_bg = adot / a if a > 1e-10 else 0.0
			delta_H_raw = last_H_eff_pos - H_bg
			tau_smooth = max(10 * dt, 1e-10)
			alpha_ema = min(dt / tau_smooth, 1.0)
			delta_H_smooth = alpha_ema * delta_H_raw + (1 - alpha_ema) * delta_H_smooth

			# Verlet position update (uses actual ȧ for this step)
			a_new = a + adot * dt + 0.5 * a_accel * dt**2

			# Evolve Friedmann backbone (no corrections)
			adot_friedmann = adot_friedmann + a_accel * dt

			# Apply correction: actual ȧ = Friedmann ȧ + a × δH
			# This is non-cumulative — adot_friedmann carries only
			# Friedmann evolution, and δH is measured fresh each step.
			H_friedmann = adot_friedmann / a_new if a_new > 1e-10 else 0.0
			H_corrected = H_friedmann + delta_H_smooth
			adot_new = a_new * max(H_corrected, 0.0)
		else:
			# Standard Friedmann: Verlet for a, a_new = a + ȧdt + ½ädt²
			a_new = a + adot * dt + 0.5 * a_accel * dt**2
			adot_new = adot + a_accel * dt

		a = max(a_new, 1e-10)
		adot = adot_new

		# ── Periodic boundary wrapping ────────────────────────────────────
		positions = positions % L

		# ── New accelerations ─────────────────────────────────────────────
		accels = pm_accelerations(positions, masses, config.G, L, config.n_grid)

		# ── Half-kick: p += ½ dt × F_pert / a ──────────────────────────
		momenta = momenta + 0.5 * dt * accels / a

		t_sim += dt

		# ── Measure effective Hubble rate (backreaction diagnostic) ───────
		last_H_eff_pos, last_H_eff_neg = _measure_effective_hubble(
			positions, momenta, masses, a, adot, L,
			config.backreaction_n_sample, backreaction_rng,
		)
		# Integrate a_eff: exact Friedmann growth × backreaction correction.
		# Using (a/a_old) as base eliminates the systematic timing bias from
		# evaluating H at the post-step state.
		H_bg = adot / a if a > 1e-10 else 0.0
		delta_H_pos = last_H_eff_pos - H_bg
		a_eff *= (a / a_old) * (1.0 + delta_H_pos * dt)

		# ── Record diagnostics ────────────────────────────────────────────
		if step % record_every == 0 or step == n_steps - 1:
			H = adot / a if a > 0 else 0.0
			history['time'].append(t_sim)
			history['scale_factor'].append(float(a))
			history['hubble_rate'].append(float(H))

			# Snapshot positions (downsampled if needed)
			pos_np = np.array(positions)
			history['positions'].append(pos_np)

			# Separation metric — use particle grid scale (n_per_dim) for
			# accurate measurement; PM grid (n_grid) has too few particles
			# per cell, so shot noise obscures the perturbation signal.
			masses_np = np.array(masses)
			corr = measure_separation(pos_np, masses_np, L, config.n_per_dim)
			history['cross_correlation'].append(corr)

			# Backreaction diagnostics
			history['effective_hubble_pos'].append(float(last_H_eff_pos))
			history['effective_hubble_neg'].append(float(last_H_eff_neg))
			history['effective_scale_factor'].append(float(a_eff))

			# Power spectra (less frequent — every 5 recordings)
			rec_idx = step // record_every
			if rec_idx % 5 == 0:
				pos_mask = labels_np == 0
				neg_mask = labels_np == 1
				k_bins, pk_p = measure_power_spectrum(
					pos_np, masses_np, L, config.n_grid, mask=pos_mask
				)
				_, pk_n = measure_power_spectrum(
					pos_np, masses_np, L, config.n_grid, mask=neg_mask
				)
				history['pk_pos'].append(pk_p)
				history['pk_neg'].append(pk_n)
				if history['k_bins'] is None:
					history['k_bins'] = k_bins

		# ── Progress ──────────────────────────────────────────────────────
		if step % max(n_steps // 10, 1) == 0 and step > 0:
			H = adot / a if a > 0 else 0.0
			delta_H = last_H_eff_pos - H if H > 0 else 0.0
			elapsed = time.time() - t_start
			print(f"  Step {step}/{n_steps}  t={t_sim:.3f} Gyr  a={a:.4f}  "
				  f"H={H:.5f} Gyr⁻¹  δH={delta_H:+.5f}  "
				  f"corr={history['cross_correlation'][-1]:+.3f}  "
				  f"[{elapsed:.1f}s]")

		# ── Stop if we've reached a_final ─────────────────────────────────
		if a >= a_final:
			# Force a final recording
			if step % record_every != 0:
				H = adot / a if a > 0 else 0.0
				history['time'].append(t_sim)
				history['scale_factor'].append(float(a))
				history['hubble_rate'].append(float(H))
				pos_np = np.array(positions)
				history['positions'].append(pos_np)
				masses_np = np.array(masses)
				corr = measure_separation(pos_np, masses_np, L, config.n_per_dim)
				history['cross_correlation'].append(corr)
				history['effective_hubble_pos'].append(float(last_H_eff_pos))
				history['effective_hubble_neg'].append(float(last_H_eff_neg))
				history['effective_scale_factor'].append(float(a_eff))
			print(f"  Reached a_final={a_final:.2f} at step {step}, t={t_sim:.3f} Gyr")
			break

	# ── Final summary ────────────────────────────────────────────────────────
	elapsed = time.time() - t_start
	H_final = adot / a if a > 0 else 0.0
	print(f"\nSimulation complete in {elapsed:.1f}s")
	print(f"  Final: t={t_sim:.3f} Gyr, a={a:.4f}, H={H_final:.5f} Gyr⁻¹")
	print(f"  Cross-correlation: {history['cross_correlation'][-1]:+.4f}")
	print(f"  Scale factor grew: {config.a_initial:.4f} → {a:.4f} "
		  f"({a / config.a_initial:.2f}×)")
	delta_H_final = last_H_eff_pos - H_final
	print(f"  Effective Hubble (pos): {last_H_eff_pos:.5f} Gyr⁻¹ (δH={delta_H_final:+.5f})")
	print(f"  Effective scale factor: {a_eff:.4f} (vs Friedmann a={a:.4f})")

	# Convert lists to arrays
	for key in ['time', 'scale_factor', 'hubble_rate', 'cross_correlation',
				'effective_hubble_pos', 'effective_hubble_neg',
				'effective_scale_factor']:
		history[key] = np.array(history[key])

	return history
