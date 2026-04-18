"""
Phase 3: Cosmological simulation initial conditions.

Generates a near-uniform particle distribution in a periodic comoving box with
small density perturbations. Both positive and negative mass particles are
placed on displaced grids (Zel'dovich-like) so structure can grow from the
perturbations self-consistently.

The perturbation spectrum uses P(k) ∝ k^n_s with a Gaussian random field,
giving scale-invariant primordial fluctuations.

Units: kpc, M☉, Gyr.
"""

import numpy as np
from dataclasses import dataclass

from antiengine.units import G_kpc_Msun_Gyr


@dataclass
class CosmoConfig:
	"""Configuration for the cosmological simulation."""

	# ── Box and grid ─────────────────────────────────────────────────────────
	box_size: float = 200_000.0     # comoving box side length (kpc) — 200 Mpc
	n_grid: int = 64                # PM grid cells per dimension

	# ── Particles ────────────────────────────────────────────────────────────
	# Particles per species are set by n_per_dim³ (placed on a uniform grid,
	# then displaced). Total particles = 2 × n_per_dim³.
	n_per_dim: int = 32             # particles per dim per species (32³ = 32768)
	neg_mass_ratio: float = 1.0     # |total neg mass| / total pos mass (1.0 = 50/50)

	# ── Cosmological parameters ─────────────────────────────────────────────
	H0: float = 0.07159            # Hubble constant in Gyr⁻¹ (70 km/s/Mpc)
	rho_crit: float = 136.0        # critical density M☉/kpc³


	# ── Initial conditions ───────────────────────────────────────────────────
	a_initial: float = 0.02        # starting scale factor
	a_final: float = 1.0           # stop when a reaches this value (1.0 = present day)
	perturbation_amplitude: float = 2e-2  # δρ/ρ amplitude for positive mass
	neg_perturbation_amplitude: float = None  # δρ/ρ for neg mass (None = same as pos)
	anti_correlated: bool = False  # if True, neg perturbations are inverted (pre-separated ICs)
	spectral_index: float = 1.0    # n_s — primordial power spectrum slope

	# ── Backreaction ─────────────────────────────────────────────────────────
	# When True, the scale factor a(t) is driven by the effective Hubble rate
	# measured from positive-mass particle pair dynamics, instead of the
	# homogeneous Friedmann equation. This captures the backreaction from
	# species segregation — the shell-theorem expansion effect that the
	# analytical Friedmann misses at 50/50. The effective a_eff(t) is always
	# computed as a diagnostic; this flag controls whether it also drives the
	# actual dynamics (kick/drift factors).
	backreaction: bool = False
	backreaction_n_sample: int = 200  # particles to subsample for H_eff measurement

	# ── Integration ──────────────────────────────────────────────────────────

	G: float = G_kpc_Msun_Gyr      # gravitational constant

	# ── Random seed ──────────────────────────────────────────────────────────
	seed: int = 42

	def __post_init__(self):
		"""Derive computed quantities."""
		self.n_particles = self.n_per_dim ** 3  # per species
		self.n_total = 2 * self.n_particles

		# Total mass in the box at a=1 (comoving volume is fixed)
		# Use critical density × box volume as the total mass budget
		box_volume = self.box_size ** 3
		total_mass = self.rho_crit * box_volume

		# Split between positive and negative mass
		# neg_mass_ratio = |M_neg| / M_pos; M_pos + |M_neg| = total_mass
		self.pos_total_mass = total_mass / (1.0 + self.neg_mass_ratio)
		self.neg_total_mass = total_mass * self.neg_mass_ratio / (1.0 + self.neg_mass_ratio)

		self.pos_particle_mass = self.pos_total_mass / self.n_particles
		self.neg_particle_mass = self.neg_total_mass / self.n_particles

		# Cell size
		self.cell_size = self.box_size / self.n_grid


def _generate_perturbation_field(n_grid, box_size, amplitude, spectral_index, rng):
	"""
	Generate a Gaussian random density perturbation field with P(k) ∝ k^n_s.

	Returns the real-space displacement potential ψ(x) such that
	the Zel'dovich displacement is  s = -∇ψ  and  δρ/ρ = -∇·s.

	The amplitude parameter controls the RMS of δρ/ρ.
	"""
	# Generate white noise in Fourier space
	shape = (n_grid, n_grid, n_grid)
	noise_real = rng.standard_normal(shape)
	noise_imag = rng.standard_normal(shape)
	noise_k = noise_real + 1j * noise_imag

	# Wavenumber magnitudes
	freqs = np.fft.fftfreq(n_grid, d=box_size / n_grid)
	kx, ky, kz = np.meshgrid(freqs, freqs, freqs, indexing='ij')
	k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

	# Power spectrum P(k) ∝ k^n_s (avoid k=0)
	k_mag_safe = np.where(k_mag > 0, k_mag, 1.0)
	pk_sqrt = np.where(k_mag > 0, k_mag_safe ** (spectral_index / 2.0), 0.0)

	# Displacement potential in Fourier space: ψ_k = δ_k / k²
	# where δ_k = P(k)^(1/2) × noise
	delta_k = pk_sqrt * noise_k
	delta_k[0, 0, 0] = 0.0  # zero mean

	# Normalise to desired amplitude
	delta_real = np.fft.ifftn(delta_k).real
	current_rms = np.std(delta_real)
	if current_rms > 0:
		delta_k *= amplitude / current_rms

	# Displacement potential: ψ_k = -δ_k / k² (Zel'dovich)
	k2 = kx**2 + ky**2 + kz**2
	k2_safe = np.where(k2 > 0, k2, 1.0)
	psi_k = np.where(k2 > 0, -delta_k / k2_safe, 0.0)

	return psi_k


def _solve_growth_factor(a_eval, omega_eff=0.0, omega_k=1.0):
	"""
	Solve the generalized anti-universe linear perturbation growth equation.

	General form (primes = d/da):
	  δ'' + P(a) δ' = Q(a) δ

	where:
	  P(a) = [3Ω_eff + 4aΩ_k] / [2a(Ω_eff + aΩ_k)]
	  Q(a) = (3/2) / [a²(Ω_eff + aΩ_k)]

	Derived from δ̈ + 2Hδ̇ = 4πGρ_crit/a³ × δ converted to d/da using
	H²(a) = H₀²(Ω_eff/a³ + Ω_k/a²) and ä = -H₀²Ω_eff/(2a²).

	For R=1 (Ω_eff=0, Ω_k=1): reduces to δ'' + (2/a)δ' = (3/2)/a³ × δ.
	For R>1: bounce at a_bounce = -Ω_eff/Ω_k where H→0; integration starts
	just above the bounce with asymptotic ICs.

	The source coefficient 3/2 = 4πGρ_crit/H₀² is ratio-independent because
	both species contribute perturbation forces in the anti-correlated
	configuration (attraction to pos overdensity + cavity repulsion from neg
	underdensity): ρ_source = ρ_pos + ρ_neg = ρ_crit.

	Returns (D_norm, f) where:
	  D_norm = D(a_eval) / D(1) — growth factor normalized to present day
	  f = a × D'(a) / D(a)     — logarithmic growth rate at a_eval
	"""
	from scipy.integrate import solve_ivp

	# Bounce scale (exists when Ω_eff < 0, i.e. R > 1)
	has_bounce = omega_eff < 0 and omega_k > 0
	a_bounce = -omega_eff / omega_k if has_bounce else 0.0

	def growth_ode(a, y):
		delta, delta_prime = y
		denom = omega_eff + a * omega_k  # proportional to H²a/H₀²
		if abs(denom) < 1e-15:
			return [delta_prime, 0.0]
		P = (3.0 * omega_eff + 4.0 * a * omega_k) / (2.0 * a * denom)
		Q = 1.5 / (a**2 * denom)
		delta_pp = Q * delta - P * delta_prime
		return [delta_prime, delta_pp]

	if has_bounce:
		# Start slightly above the bounce where the ODE is singular (H→0).
		# Asymptotic IC: at the bounce the leading-order balance gives
		# δ' = 3/(a_b² Ω_k) × δ (from the limit as the ȧ² δ'' term vanishes).
		eps = max(1e-4, a_bounce * 1e-4)
		a_start = a_bounce + eps
		delta_prime_0 = 3.0 / (a_bounce**2 * omega_k)
	else:
		a_start = 1e-4
		delta_prime_0 = 0.0

	a_end = max(20.0, a_eval * 1.5)

	sol = solve_ivp(
		growth_ode, [a_start, a_end], [1.0, delta_prime_0],
		method='DOP853', rtol=1e-12, atol=1e-15,
		dense_output=True, max_step=0.01,
	)

	D_eval, Dp_eval = sol.sol(a_eval)
	D_1, _ = sol.sol(1.0)

	D_norm = D_eval / D_1
	f = a_eval * Dp_eval / D_eval

	return D_norm, f


def _zeldovich_displacements(psi_k, box_size, n_grid):
	"""
	Compute Zel'dovich displacement vectors from the displacement potential.

	s(x) = -∇ψ(x) in real space, computed via Fourier derivatives.

	Returns (n_grid, n_grid, n_grid, 3) displacement field.
	"""
	freqs = np.fft.fftfreq(n_grid, d=box_size / n_grid)
	kx, ky, kz = np.meshgrid(freqs, freqs, freqs, indexing='ij')

	# s_k = -i·k·ψ_k (gradient in Fourier space)
	sx_k = -1j * 2 * np.pi * kx * psi_k
	sy_k = -1j * 2 * np.pi * ky * psi_k
	sz_k = -1j * 2 * np.pi * kz * psi_k

	sx = np.fft.ifftn(sx_k).real
	sy = np.fft.ifftn(sy_k).real
	sz = np.fft.ifftn(sz_k).real

	return np.stack([sx, sy, sz], axis=-1)


def initialize_cosmological(config: CosmoConfig):
	"""
	Generate initial conditions for the cosmological simulation.

	Both species start on jittered grids and are displaced by Zel'dovich
	perturbation fields. Three modes are supported:

	1. Shared field (default): both species get the same perturbation pattern,
	   optionally at different amplitudes. Starts correlated (r ≈ +1).

	2. Anti-correlated (anti_correlated=True): negative mass gets the INVERTED
	   perturbation field. Where positive mass has an overdensity, negative
	   mass has an underdensity and vice versa. Starts anti-correlated (r ≈ -1).
	   This represents the pre-separated state after the rapid early-universe
	   separation process, with the species already in complementary distributions.

	Returns:
		positions : (N_total, 3) comoving positions in [0, box_size)
		velocities: (N_total, 3) peculiar velocities (comoving momentum / a²)
		masses    : (N_total,)   signed masses (+pos, -neg)
		labels    : (N_total,)   0=positive, 1=negative
	"""
	rng = np.random.default_rng(config.seed)
	n = config.n_per_dim
	L = config.box_size

	# ── Uniform grid positions ───────────────────────────────────────────────
	cell = L / n
	ix, iy, iz = np.meshgrid(
		np.arange(n), np.arange(n), np.arange(n), indexing='ij'
	)
	grid_pos = np.stack([
		(ix.ravel() + 0.5) * cell,
		(iy.ravel() + 0.5) * cell,
		(iz.ravel() + 0.5) * cell,
	], axis=-1)  # (n³, 3)

	# Break grid coherence with small random jitter (prevents "grid memory"
	# artifacts where the regular grid structure persists through the
	# simulation, especially visible in the negative mass density field).
	# Jitter is ±0.15 cells — enough to destroy grid periodicity while keeping
	# shot noise below the perturbation signal on scales > a few cells.
	pos_jitter = (rng.random((n**3, 3)) - 0.5) * 0.3 * cell
	pos_grid = (grid_pos + pos_jitter) % L

	if config.anti_correlated:
		# Independent jitter for neg particles so their shot-noise pattern
		# is decoupled from pos mass. At z=49 the Zel'dovich displacements
		# are negligible vs particle spacing, so CIC density is dominated
		# by where particles sit — shared jitter → r ≈ +1 regardless of
		# the inverted displacement field.
		neg_jitter = (rng.random((n**3, 3)) - 0.5) * 0.3 * cell
		neg_grid = (grid_pos + neg_jitter) % L
	else:
		neg_grid = pos_grid

	# ── Generate perturbation field ──────────────────────────────────────────
	# Generate the spatial pattern (unit amplitude) and scale per species
	psi_k = _generate_perturbation_field(
		n, L, 1.0, config.spectral_index, rng
	)
	disp_field = _zeldovich_displacements(psi_k, L, n)  # (n, n, n, 3)
	displacements_unit = disp_field.reshape(-1, 3)  # (n³, 3) — unit amplitude

	# Positive mass displacements
	pos_amp = config.perturbation_amplitude

	# Negative mass amplitude
	neg_amp = config.neg_perturbation_amplitude
	if neg_amp is None:
		neg_amp = pos_amp

	# Friedmann parameters for this mass ratio:
	# Ω_eff = (1-R)/(1+R), Ω_k = 2R/(1+R)
	# These determine H(a), ä(a), and the growth factor ODE coefficients.
	R = config.neg_mass_ratio
	omega_eff = (1.0 - R) / (1.0 + R)
	omega_k = 2.0 * R / (1.0 + R)

	if config.anti_correlated:
		# Compute the anti-universe growth factor D(a) normalized to D(1)=1.
		# perturbation_amplitude represents δρ/ρ at a=1 (present day).
		# At a_initial, the displacement is scaled by D(a_initial)/D(1).
		# This makes the result independent of a_initial (in the linear regime).
		D_norm, f_growth = _solve_growth_factor(config.a_initial, omega_eff, omega_k)
		pos_displacements = displacements_unit * (pos_amp * D_norm)
		neg_displacements = -displacements_unit * (neg_amp * D_norm)
	else:
		# Standard mode: both species correlated (net force suppressed at 50/50)
		# Use D ∝ a as a rough approximation (growth is slow when forces cancel)
		D_norm = config.a_initial
		f_growth = 1.0
		pos_displacements = displacements_unit * (D_norm * pos_amp)
		neg_displacements = displacements_unit * (D_norm * neg_amp)

	# ── Apply displacements to both species ──────────────────────────────
	pos_positions = pos_grid + pos_displacements
	neg_positions = neg_grid + neg_displacements

	# Wrap into periodic box
	pos_positions = pos_positions % L
	neg_positions = neg_positions % L

	# ── Zel'dovich velocities ────────────────────────────────────────────────
	# Peculiar velocity: v = H(a) × f(a) × displacement
	# where f = a×D'/D is the logarithmic growth rate.
	# H(a) from the first Friedmann equation (reduces to H₀/a for Milne).
	a_i = config.a_initial
	H2_at_a = config.H0**2 * (omega_eff / a_i**3 + omega_k / a_i**2)
	if H2_at_a < 0:
		raise ValueError(
			f"H²(a₀) < 0: a_initial={a_i} is below the bounce at "
			f"a_bounce={-omega_eff/omega_k:.4f}. Increase a_initial."
		)
	H_at_a = np.sqrt(H2_at_a)
	pos_velocities = H_at_a * f_growth * pos_displacements
	neg_velocities = H_at_a * f_growth * neg_displacements

	# ── Combine arrays ───────────────────────────────────────────────────────
	n_part = config.n_particles

	positions = np.concatenate([pos_positions, neg_positions], axis=0)
	velocities = np.concatenate([pos_velocities, neg_velocities], axis=0)

	masses = np.concatenate([
		np.full(n_part, config.pos_particle_mass),
		np.full(n_part, -config.neg_particle_mass),
	])

	labels = np.concatenate([
		np.zeros(n_part, dtype=np.int32),
		np.ones(n_part, dtype=np.int32),
	])

	# ── Summary ──────────────────────────────────────────────────────────────
	mode = "anti-correlated" if config.anti_correlated else "shared"
	print(f"Cosmological IC summary:")
	print(f"  Box size:         {L:.0f} kpc ({L/1000:.0f} Mpc)")
	print(f"  Grid:             {config.n_grid}³ = {config.n_grid**3} cells")
	print(f"  Particles/species:{n_part} ({n}³)")
	print(f"  Total particles:  {2 * n_part}")
	print(f"  Pos particle mass:{config.pos_particle_mass:.3e} M☉")
	print(f"  Neg particle mass:{config.neg_particle_mass:.3e} M☉")
	print(f"  neg_mass_ratio:   {config.neg_mass_ratio} (|M-|/M+)")
	print(f"  a_initial:        {config.a_initial}")
	print(f"  D(a_init)/D(1):   {D_norm:.6f}")
	print(f"  f(a_init):        {f_growth:.4f}")
	print(f"  δρ/ρ at a=1 (pos):{pos_amp}")
	print(f"  δρ/ρ at a=1 (neg):{neg_amp}")
	print(f"  IC mode:          {mode}")
	print(f"  H₀:              {config.H0:.5f} Gyr⁻¹ ({config.H0 * 977.8:.1f} km/s/Mpc)")

	return positions, velocities, masses, labels
