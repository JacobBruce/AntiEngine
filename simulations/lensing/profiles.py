"""
3D density profiles and their line-of-sight projected surface mass densities Σ(R).

All profiles are spherically symmetric and defined in physical units (kpc, M☉).
Each profile class provides:
	rho(r)          — 3D mass density [M☉/kpc³]
	M_enc(r)        — enclosed mass within sphere of radius r [M☉]
	sigma(R)        — projected surface mass density [M☉/kpc²]
	mean_sigma(R)   — mean Σ inside projected radius R: ⟨Σ⟩(<R)
	delta_sigma(R)  — excess surface density: ΔΣ(R) = ⟨Σ⟩(<R) − Σ(R)

All methods accept and return plain Python floats for scalar input,
or numpy arrays for array input. This ensures compatibility with
scipy.integrate.quad which requires float-returning callbacks.
"""

import numpy as np
from scipy import integrate
from scipy.special import betainc as sp_betainc


def _to_array(x):
	"""Convert to float64 array; return (array, was_scalar)."""
	x = np.asarray(x, dtype=float)
	scalar = x.ndim == 0
	return np.atleast_1d(x), scalar


def _from_array(arr, scalar):
	"""Return Python float if input was scalar, else numpy array."""
	return arr.item() if scalar else arr


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class MassProfile:
	"""Abstract base for a spherically symmetric mass profile."""

	def __init__(self, label: str = ""):
		self.label = label

	def rho(self, r):
		"""3D density at radius r [M☉/kpc³]."""
		raise NotImplementedError

	def M_enc(self, r):
		"""Enclosed mass within radius r [M☉]."""
		raise NotImplementedError

	def sigma(self, R):
		"""Projected surface mass density at projected radius R [M☉/kpc²]."""
		raise NotImplementedError

	def mean_sigma(self, R):
		"""Mean projected surface density inside circle of radius R [M☉/kpc²]."""
		R, scalar = _to_array(R)
		result = np.zeros_like(R)
		for i, Ri in enumerate(R):
			if Ri <= 0:
				continue
			# ⟨Σ⟩(<R) = (2/R²) ∫₀ᴿ Σ(R') R' dR'
			val, _ = integrate.quad(
				lambda Rp: self.sigma(Rp) * Rp, 0, float(Ri),
				limit=200, epsrel=1e-8,
			)
			result[i] = 2.0 * val / (Ri ** 2)
		return _from_array(result, scalar)

	def delta_sigma(self, R):
		"""Excess surface density: ΔΣ(R) = ⟨Σ⟩(<R) − Σ(R) [M☉/kpc²].

		This is the direct observable in galaxy-galaxy weak lensing:
			ΔΣ(R) = Σ_crit × γ_t(R)
		"""
		R, scalar = _to_array(R)
		ms = np.atleast_1d(np.asarray(self.mean_sigma(R), dtype=float))
		s = np.atleast_1d(np.asarray(self.sigma(R), dtype=float))
		return _from_array(ms - s, scalar)


# ─────────────────────────────────────────────────────────────────────────────
# NFW profile
# ─────────────────────────────────────────────────────────────────────────────

class NFWProfile(MassProfile):
	"""
	Navarro-Frenk-White (1996) profile.

	ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]

	Parameters
	----------
	M_200   : virial mass [M☉] — mass inside r_200 (200× critical density)
	c       : concentration parameter c = r_200 / r_s
	rho_crit: critical density of the universe [M☉/kpc³]
	"""

	def __init__(self, M_200, c, rho_crit=136.0, label="NFW"):
		super().__init__(label)
		self.M_200 = M_200
		self.c = c
		self.rho_crit = rho_crit

		# r_200: sphere enclosing 200 × ρ_crit
		self.r_200 = (3 * M_200 / (4 * np.pi * 200 * rho_crit)) ** (1.0 / 3.0)
		self.r_s = self.r_200 / c

		# Characteristic overdensity
		self.rho_s = M_200 / (4 * np.pi * self.r_s ** 3 *
			(np.log(1 + c) - c / (1 + c)))

	def rho(self, r):
		r, scalar = _to_array(r)
		x = r / self.r_s
		return _from_array(self.rho_s / (x * (1 + x) ** 2), scalar)

	def M_enc(self, r):
		r, scalar = _to_array(r)
		x = r / self.r_s
		return _from_array(
			4 * np.pi * self.rho_s * self.r_s ** 3 * (np.log(1 + x) - x / (1 + x)),
			scalar,
		)

	def sigma(self, R):
		"""Analytical NFW surface density (Bartelmann 1996, Wright & Brainerd 2000)."""
		R, scalar = _to_array(R)
		x = R / self.r_s
		result = np.zeros_like(x)

		lo = x < 1.0
		xm = x[lo]
		result[lo] = (1.0 / (xm ** 2 - 1)) * (
			1.0 - np.arccosh(1.0 / xm) / np.sqrt(1.0 - xm ** 2)
		)

		eq = np.abs(x - 1.0) < 1e-6
		result[eq] = 1.0 / 3.0

		hi = x > 1.0
		xm = x[hi]
		result[hi] = (1.0 / (xm ** 2 - 1)) * (
			1.0 - np.arccos(1.0 / xm) / np.sqrt(xm ** 2 - 1.0)
		)

		return _from_array(2.0 * self.rho_s * self.r_s * result, scalar)

	def mean_sigma(self, R):
		"""Analytical ⟨Σ⟩(<R) for NFW (Wright & Brainerd 2000)."""
		R, scalar = _to_array(R)
		x = R / self.r_s
		result = np.zeros_like(x)

		lo = x < 1.0
		xm = x[lo]
		result[lo] = (
			np.arccosh(1.0 / xm) / np.sqrt(1.0 - xm ** 2)
			+ np.log(xm / 2.0)
		)

		eq = np.abs(x - 1.0) < 1e-6
		result[eq] = 1.0 + np.log(0.5)

		hi = x > 1.0
		xm = x[hi]
		result[hi] = (
			np.arccos(1.0 / xm) / np.sqrt(xm ** 2 - 1.0)
			+ np.log(xm / 2.0)
		)

		return _from_array(4.0 * self.rho_s * self.r_s * result / (x ** 2), scalar)


# ─────────────────────────────────────────────────────────────────────────────
# Burkert profile (cored)
# ─────────────────────────────────────────────────────────────────────────────

class BurkertProfile(MassProfile):
	"""
	Burkert (1995) profile — a cored dark matter profile.

	ρ(r) = ρ_0 / [(1 + r/r_0)(1 + (r/r_0)²)]

	Favoured by dwarf galaxy observations. Has a constant-density core (no cusp).

	Parameters
	----------
	rho_0 : central density [M☉/kpc³]
	r_0   : core radius [kpc]
	r_max : truncation radius [kpc] (for projection integrals)
	"""

	def __init__(self, rho_0, r_0, r_max=500.0, label="Burkert"):
		super().__init__(label)
		self.rho_0 = rho_0
		self.r_0 = r_0
		self.r_max = r_max

	def rho(self, r):
		r, scalar = _to_array(r)
		x = r / self.r_0
		return _from_array(self.rho_0 / ((1 + x) * (1 + x ** 2)), scalar)

	def M_enc(self, r):
		r, scalar = _to_array(r)
		x = r / self.r_0
		return _from_array(
			np.pi * self.rho_0 * self.r_0 ** 3 * (
				np.log(1 + x ** 2) + 2 * np.log(1 + x) - 2 * np.arctan(x)
			), scalar,
		)

	def sigma(self, R):
		"""Numerical LOS projection."""
		R, scalar = _to_array(R)
		result = np.zeros_like(R)
		for i, Ri in enumerate(R):
			Ri_f = float(Ri)
			if Ri_f >= self.r_max:
				continue
			z_max = np.sqrt(self.r_max ** 2 - Ri_f ** 2)
			val, _ = integrate.quad(
				lambda z: self.rho(np.sqrt(Ri_f ** 2 + z ** 2)),
				0, z_max, limit=200, epsrel=1e-8,
			)
			result[i] = 2.0 * val
		return _from_array(result, scalar)


# ─────────────────────────────────────────────────────────────────────────────
# Pseudo-isothermal profile (cored)
# ─────────────────────────────────────────────────────────────────────────────

class PseudoIsothermalProfile(MassProfile):
	"""
	Pseudo-isothermal sphere — a cored isothermal profile.

	ρ(r) = ρ_0 / [1 + (r/r_c)²]

	Produces perfectly flat rotation curves at r >> r_c.

	Parameters
	----------
	rho_0 : central density [M☉/kpc³]
	r_c   : core radius [kpc]
	r_max : truncation radius [kpc]
	"""

	def __init__(self, rho_0, r_c, r_max=500.0, label="Pseudo-isothermal"):
		super().__init__(label)
		self.rho_0 = rho_0
		self.r_c = r_c
		self.r_max = r_max

	def rho(self, r):
		r, scalar = _to_array(r)
		return _from_array(self.rho_0 / (1 + (r / self.r_c) ** 2), scalar)

	def M_enc(self, r):
		r, scalar = _to_array(r)
		x = r / self.r_c
		return _from_array(
			4 * np.pi * self.rho_0 * self.r_c ** 3 * (x - np.arctan(x)), scalar,
		)

	def sigma(self, R):
		"""Analytical Σ(R) for truncated pseudo-isothermal sphere."""
		R, scalar = _to_array(R)
		result = np.zeros_like(R)
		mask = R < self.r_max
		Rm = R[mask]
		z_max = np.sqrt(self.r_max ** 2 - Rm ** 2)
		s = np.sqrt(Rm ** 2 + self.r_c ** 2)
		result[mask] = 2.0 * self.rho_0 * self.r_c ** 2 * np.arctan(z_max / s) / s
		return _from_array(result, scalar)


# ─────────────────────────────────────────────────────────────────────────────
# Exponential disk (baryonic component)
# ─────────────────────────────────────────────────────────────────────────────

class ExponentialDiskProfile(MassProfile):
	"""
	Infinitely thin exponential disk (face-on projection).

	Σ(R) = (M_d / 2π r_d²) × exp(−R / r_d)

	This is the baryonic component commonly used in decompositions.

	Parameters
	----------
	M_d : total disk mass [M☉]
	r_d : scale radius [kpc]
	"""

	def __init__(self, M_d, r_d, label="Exponential disk"):
		super().__init__(label)
		self.M_d = M_d
		self.r_d = r_d
		self.Sigma_0 = M_d / (2 * np.pi * r_d ** 2)

	def rho(self, r):
		raise NotImplementedError("Thin disk has no well-defined 3D density. Use sigma().")

	def M_enc(self, r):
		"""Enclosed mass within projected radius R (2D cumulative)."""
		r, scalar = _to_array(r)
		x = r / self.r_d
		return _from_array(self.M_d * (1 - (1 + x) * np.exp(-x)), scalar)

	def sigma(self, R):
		R, scalar = _to_array(R)
		return _from_array(self.Sigma_0 * np.exp(-R / self.r_d), scalar)


# ─────────────────────────────────────────────────────────────────────────────
# Hernquist bulge (baryonic component)
# ─────────────────────────────────────────────────────────────────────────────

class HernquistProfile(MassProfile):
	"""
	Hernquist (1990) profile for bulge / stellar component.

	ρ(r) = M a / (2π r (r+a)³)

	Parameters
	----------
	M : total mass [M☉]
	a : scale radius [kpc]
	"""

	def __init__(self, M, a, label="Hernquist"):
		super().__init__(label)
		self.M = M
		self.a = a

	def rho(self, r):
		r, scalar = _to_array(r)
		return _from_array(self.M * self.a / (2 * np.pi * r * (r + self.a) ** 3), scalar)

	def M_enc(self, r):
		r, scalar = _to_array(r)
		return _from_array(self.M * r ** 2 / (r + self.a) ** 2, scalar)

	def sigma(self, R):
		"""Analytical Σ(R) for Hernquist (Hernquist 1990, eq. 33)."""
		R, scalar = _to_array(R)
		s = R / self.a
		result = np.zeros_like(R)

		lo = s < 1.0
		sm = s[lo]
		sm2 = sm ** 2
		result[lo] = (
			(2 + sm2) * np.arccosh(1.0 / sm) / (1 - sm2) ** 1.5
			- 1.0 / (1 - sm2)
		)

		eq = np.abs(s - 1.0) < 1e-6
		result[eq] = 2.0 / 3.0

		hi = s > 1.0
		sm = s[hi]
		sm2 = sm ** 2
		result[hi] = (
			(2 + sm2) * np.arccos(1.0 / sm) / (sm2 - 1) ** 1.5
			- 1.0 / (sm2 - 1)
		)

		return _from_array(self.M / (2 * np.pi * self.a ** 2) * result, scalar)


# ─────────────────────────────────────────────────────────────────────────────
# Point mass (SMBH / compact object)
# ─────────────────────────────────────────────────────────────────────────────

class PointMassProfile(MassProfile):
	"""
	Point mass profile (e.g. central SMBH).

	M_enc(r) = M for all r > 0.
	Σ(R) = 0 for R > 0 (all mass at the origin).
	ΔΣ(R) = M / (π R²) — analytical, no numerical integration needed.

	Parameters
	----------
	M : total mass [M☉]
	"""

	def __init__(self, M, label="Point mass"):
		super().__init__(label)
		self.M = M

	def rho(self, r):
		r, scalar = _to_array(r)
		return _from_array(np.zeros_like(r), scalar)

	def M_enc(self, r):
		r, scalar = _to_array(r)
		return _from_array(np.where(r > 0, self.M, 0.0), scalar)

	def sigma(self, R):
		R, scalar = _to_array(R)
		return _from_array(np.zeros_like(R), scalar)

	def mean_sigma(self, R):
		R, scalar = _to_array(R)
		return _from_array(np.where(R > 0, self.M / (np.pi * R ** 2), 0.0), scalar)

	def delta_sigma(self, R):
		R, scalar = _to_array(R)
		return _from_array(np.where(R > 0, self.M / (np.pi * R ** 2), 0.0), scalar)


# ─────────────────────────────────────────────────────────────────────────────
# Cavity model profile (anti-universe dark matter) — density deficit
# ─────────────────────────────────────────────────────────────────────────────

class CavityProfile(MassProfile):
	"""
	Neg-mass cavity dark matter profile based on empirical 2D N-body simulations.

	In the anti-universe model, a galaxy repels surrounding negative mass,
	creating a density deficit that acts as effective positive mass.

	The deficit profile is parametrized in rescaled coordinates x = r / R_halo,
	where R_halo is the outer boundary of the halo (analogous to R_bnd in the
	2D sims). The recovery function (ρ_neg/ρ_bg) rises from ~0 at the center
	to ~1 at r = R_halo.

	Three profile types are supported:

	1. 'beta_cdf' (default) — regularized incomplete beta function I_x(a, b):

		recovery(x) = I_x(a_beta, b_beta)
		deficit(x) = 1 − I_x(a_beta, b_beta)

		Naturally confined to [0,1] on x ∈ [0,1], so the deficit vanishes
		exactly at R_halo. Best-fit to 2D cavity N-body sims (ρ_bg from
		boundary density, full profile to R_bnd):

		  R_bnd/R_eq ≈ 1.0  →  a ≈ 1.8, b ≈ 0.7
		  R_bnd/R_eq ≈ 2.0  →  a ≈ 1.5, b ≈ 0.4
		  R_bnd/R_eq ≈ 2.8  →  a ≈ 1.1, b ≈ 0.3

		Default (a_beta=1.5, b_beta=0.4): median across 11 sim configs.
		RMS residuals: 0.03–0.05 (2–3× better than stretched_exp or logistic).

	2. 'stretched_exp' — stretched exponential:

		deficit(x) = exp(−(x / R_s)^b)

		Approaches zero asymptotically (does not reach 0 at finite x).

	3. 'logistic' — sigmoid form:

		deficit(x) = R_s^α / (R_s^α + x^α)

		Approaches zero asymptotically (does not reach 0 at finite x).

	ρ_bg is determined self-consistently so that the enclosed deficit mass
	within R_halo equals M_DM:

		M_DM = 4π ρ_bg R_halo³ ∫₀¹ s² × deficit(s) ds

	Parameters
	----------
	M_DM         : total effective DM mass within R_halo [M☉]
	R_halo       : halo boundary / transition radius [kpc]
	profile_type : 'beta_cdf', 'stretched_exp', or 'logistic'
	a_beta       : beta CDF shape parameter a (beta_cdf only, default 1.5)
	b_beta       : beta CDF shape parameter b (beta_cdf only, default 0.4)
	alpha        : logistic steepness (logistic only, default 5.5)
	R_s          : scale parameter in units of R_halo (stretched_exp/logistic)
	b            : stretched exp exponent (stretched_exp only, default 3.0)
	r_max        : maximum radius for LOS integration [kpc]
	"""

	def __init__(self, M_DM, R_halo, profile_type='beta_cdf',
				 a_beta=1.5, b_beta=0.4,
				 alpha=5.5, R_s=None, b=3.0,
				 r_max=2000.0, label="Cavity"):
		super().__init__(label)
		self.M_DM = M_DM
		self.R_halo = R_halo
		self.profile_type = profile_type
		self.a_beta = a_beta
		self.b_beta = b_beta
		self.alpha = alpha
		self.b = b
		self.r_max = r_max

		# Set default R_s based on profile type
		if R_s is None:
			self.R_s = 0.52 if profile_type == 'stretched_exp' else 0.45
		else:
			self.R_s = R_s

		# Derive ρ_bg such that M_enc(R_halo) = M_DM
		I, _ = integrate.quad(
			lambda s: s ** 2 * self._deficit_at(s),
			0, 1.0, limit=200, epsrel=1e-10,
		)
		self._deficit_integral = I
		self.rho_bg = M_DM / (4.0 * np.pi * R_halo ** 3 * I)

	def _deficit_at(self, x):
		"""Deficit function (ρ_DM/ρ_bg) at rescaled coordinate x = r/R_halo."""
		x = np.asarray(x, dtype=float)
		if self.profile_type == 'beta_cdf':
			xc = np.clip(x, 0.0, 1.0)
			return 1.0 - sp_betainc(self.a_beta, self.b_beta, xc)
		elif self.profile_type == 'stretched_exp':
			return np.exp(-np.power(x / self.R_s, self.b))
		else:
			# Logistic: Rs^α / (Rs^α + x^α)
			Rs_a = np.power(self.R_s, self.alpha)
			return Rs_a / (Rs_a + np.power(x, self.alpha))

	def _f(self, r):
		"""Recovery function: ρ_neg(r) / ρ_bg = 1 − deficit."""
		r = np.asarray(r, dtype=float)
		x = r / self.R_halo
		return 1.0 - self._deficit_at(x)

	def rho(self, r):
		"""Effective DM density = deficit from background."""
		r, scalar = _to_array(r)
		result = self.rho_bg * (1.0 - self._f(r))
		return _from_array(result, scalar)

	def M_enc(self, r):
		"""Enclosed deficit mass (numerical)."""
		r, scalar = _to_array(r)
		result = np.zeros_like(r)
		for i, ri in enumerate(r):
			ri_f = float(ri)
			if ri_f <= 0:
				continue
			val, _ = integrate.quad(
				lambda rp: 4 * np.pi * rp ** 2 * self.rho_bg * (1.0 - self._f(rp)),
				0, ri_f, limit=200, epsrel=1e-8,
			)
			result[i] = val
		return _from_array(result, scalar)

	def sigma(self, R):
		"""Projected surface mass density (numerical LOS integration)."""
		R, scalar = _to_array(R)
		result = np.zeros_like(R)
		for i, Ri in enumerate(R):
			Ri_f = float(Ri)
			z_max = np.sqrt(max(self.r_max ** 2 - Ri_f ** 2, 0))
			if z_max <= 0:
				continue
			val, _ = integrate.quad(
				lambda z: self.rho(np.sqrt(Ri_f ** 2 + z ** 2)),
				0, z_max, limit=200, epsrel=1e-8,
			)
			result[i] = 2.0 * val
		return _from_array(result, scalar)


# ─────────────────────────────────────────────────────────────────────────────
# Composite profile (sum of multiple components)
# ─────────────────────────────────────────────────────────────────────────────

class CompositeProfile(MassProfile):
	"""Sum of multiple mass profiles (e.g., disk + bulge + halo)."""

	def __init__(self, profiles, label="Composite"):
		super().__init__(label)
		self.profiles = profiles

	def rho(self, r):
		return sum(p.rho(r) for p in self.profiles)

	def M_enc(self, r):
		return sum(p.M_enc(r) for p in self.profiles)

	def sigma(self, R):
		return sum(p.sigma(R) for p in self.profiles)

	def mean_sigma(self, R):
		return sum(p.mean_sigma(R) for p in self.profiles)

	def delta_sigma(self, R):
		return sum(p.delta_sigma(R) for p in self.profiles)
