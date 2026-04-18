"""
Milky Way Rotation Curve Decomposition

Takes observed rotation curve data and fits a parametric mass model to determine:
  1. How the galaxy's baryonic mass must be distributed
  2. How much dark matter is required and its radial profile
  3. The residual (unexplained) mass at each radius

Observational data from:
  - Eilers et al. 2019 (ApJ 871, 120): 5-25 kpc, Jeans modeling of APOGEE+Gaia
  - Sofue 2012/2015 (PASJ): Inner galaxy compilation (0.2-5 kpc)
  - Huang et al. 2016 (MNRAS 463, 2623): Extended to ~100 kpc using halo stars
  - Bhattacharjee et al. 2014: Outer halo tracers

Units: kpc, M☉, km/s throughout.
"""

import numpy as np
from scipy.optimize import minimize
from scipy import integrate
import matplotlib.pyplot as plt
from antiengine.units import G_kpc_Msun_Gyr, kpc_per_Gyr_to_kms


# ─────────────────────────────────────────────────────────────────────────────
# Observed Milky Way rotation curve data
# ─────────────────────────────────────────────────────────────────────────────

def load_mw_rotation_curve():
	"""
	Return observed MW rotation curve data from published measurements.

	Returns dict with 'r' [kpc], 'v' [km/s], 'v_err' [km/s], 'source' [str].

	Sources:
	  Sofue 2012 (PASJ 64, 75): inner galaxy, R=0.2-5 kpc
	  Eilers et al. 2019 (ApJ 871, 120): R=5-25 kpc (gold standard, ±2-3 km/s)
	  Huang et al. 2016 (MNRAS 463, 2623): R=16-100 kpc (halo K giants)

	Eilers data uses R☉=8.122 kpc, V☉=229 km/s.
	Inner data rescaled to same R☉ convention.
	"""

	# Inner galaxy from Sofue 2012/2015 compilation (representative points)
	# These are less precise due to non-circular motions from the bar
	r_inner = np.array([0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
	v_inner = np.array([170, 205, 215, 215, 220, 225, 220, 215, 225, 230])
	e_inner = np.array([ 30,  20,  15,  15,  15,  15,  15,  15,  10,  10])
	s_inner = ['Sofue'] * len(r_inner)

	# Eilers et al. 2019, Table 1 (highest precision, ±2-5 km/s systematic)
	# v_c gently declining at -1.7 ± 0.1 km/s/kpc
	r_eilers = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5,
						 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0,
						 12.5, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
						 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0])
	v_eilers = np.array([236.0, 235.0, 233.5, 232.5, 231.5, 230.5, 229.0, 228.0,
						 227.0, 226.0, 225.5, 225.0, 224.0, 223.0, 222.0,
						 221.5, 221.0, 220.0, 218.5, 217.5, 216.0, 215.0,
						 213.5, 212.5, 211.0, 210.0, 209.0, 207.5, 206.0])
	e_eilers = np.full(len(r_eilers), 3.0)  # ~3 km/s systematic floor
	s_eilers = ['Eilers2019'] * len(r_eilers)

	# Huang et al. 2016 — extended rotation curve from halo K-giants
	# Larger uncertainties at large radii
	r_huang = np.array([30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0])
	v_huang = np.array([200.0, 195.0, 192.0, 185.0, 178.0, 170.0, 163.0, 150.0])
	e_huang = np.array([  10,   12,   15,   18,   20,   22,   25,   30])
	s_huang = ['Huang2016'] * len(r_huang)

	r_all = np.concatenate([r_inner, r_eilers, r_huang])
	v_all = np.concatenate([v_inner, v_eilers, v_huang])
	e_all = np.concatenate([e_inner, e_eilers, e_huang])
	s_all = s_inner + s_eilers + s_huang

	return {
		'r': r_all,
		'v': v_all,
		'v_err': e_all,
		'source': s_all,
	}


# ─────────────────────────────────────────────────────────────────────────────
# Parametric mass models
# ─────────────────────────────────────────────────────────────────────────────

def M_enc_point(r, M):
	"""Point mass (SMBH). M_enc = M for all r > 0."""
	return np.where(r > 0, M, 0.0)


def M_enc_hernquist(r, M, a):
	"""Hernquist (1990) bulge. M_enc(r) = M × r² / (r + a)²."""
	return M * r**2 / (r + a)**2


def M_enc_exponential_disk(r, M_d, r_d):
	"""
	Exponential disk enclosed mass (2D cumulative).
	M_enc(R) = M_d × [1 - (1 + R/r_d) × exp(-R/r_d)]
	"""
	x = r / r_d
	return M_d * (1.0 - (1.0 + x) * np.exp(-x))


def M_enc_gas_disk(r, M_gas, r_gas):
	"""Gas disk — modeled as exponential with larger scale radius."""
	x = r / r_gas
	return M_gas * (1.0 - (1.0 + x) * np.exp(-x))


def M_enc_nfw(r, M_200, c, rho_crit=126.0):
	"""
	NFW enclosed mass.
	r_200 = (3 M_200 / 4π × 200 × ρ_crit)^(1/3)
	r_s = r_200 / c
	M(r) = M_200 × [ln(1+x) - x/(1+x)] / [ln(1+c) - c/(1+c)]
	"""
	r_200 = (3 * M_200 / (4 * np.pi * 200 * rho_crit))**(1.0 / 3.0)
	r_s = r_200 / c
	x = r / r_s
	f_x = np.log(1 + x) - x / (1 + x)
	f_c = np.log(1 + c) - c / (1 + c)
	return M_200 * f_x / f_c


def M_enc_burkert(r, rho_0, r_0):
	"""
	Burkert (1995) cored profile enclosed mass.
	ρ(r) = ρ_0 / [(1 + r/r_0)(1 + (r/r_0)²)]
	M(r) = π ρ_0 r_0³ [ln(1 + r/r_0) + ½ ln(1 + (r/r_0)²) - arctan(r/r_0)]
	"""
	x = r / r_0
	return np.pi * rho_0 * r_0**3 * (
		np.log(1 + x) + 0.5 * np.log(1 + x**2) - np.arctan(x)
	)


def v_circular(M_enc, r):
	"""Circular velocity from enclosed mass: v_c = sqrt(G × M_enc / r) [km/s]."""
	G = G_kpc_Msun_Gyr
	v = np.sqrt(G * np.maximum(M_enc, 0) / np.maximum(r, 1e-6))
	return v * kpc_per_Gyr_to_kms


# ─────────────────────────────────────────────────────────────────────────────
# Galaxy model
# ─────────────────────────────────────────────────────────────────────────────

# Baryonic parameter definitions (shared by all DM halo types)
BARYON_PARAM_NAMES = ['M_bulge', 'a_bulge', 'M_disk', 'r_disk', 'M_gas', 'r_gas']
BARYON_DEFAULTS = {
	'M_bulge': 1.0e10,   # M☉
	'a_bulge': 0.6,      # kpc
	'M_disk':  4.5e10,   # M☉
	'r_disk':  2.6,      # kpc
	'M_gas':   1.0e10,   # M☉
	'r_gas':   4.0,      # kpc (gas is more extended)
}
BARYON_BOUNDS = {
	'M_bulge': (5e9,  2.5e10),
	'a_bulge': (0.2,  1.5),
	'M_disk':  (2e10, 8e10),
	'r_disk':  (1.5,  4.0),
	'M_gas':   (5e9,  2e10),
	'r_gas':   (2.0,  8.0),
}

# DM halo configurations per type
DM_CONFIGS = {
	'nfw': {
		'params': ['M_200', 'c_200'],
		'defaults': {'M_200': 8.0e11, 'c_200': 12.0},
		'bounds': {'M_200': (3e11, 2e12), 'c_200': (5.0, 25.0)},
		'label': 'NFW',
	},
	'burkert': {
		'params': ['rho_0', 'r_0'],
		'defaults': {'rho_0': 2.0e7, 'r_0': 10.0},  # M☉/kpc³, kpc
		'bounds': {'rho_0': (1e5, 1e9), 'r_0': (1.0, 40.0)},
		'label': 'Burkert',
	},
	'cavity': {
		# Logistic deficit profile: deficit(x) = R_s^α / (R_s^α + x^α)
		# ρ_bg derived self-consistently: M_enc(R_halo) = cav_M_DM
		'params': ['cav_M_DM', 'cav_R_halo', 'cav_alpha', 'cav_R_s'],
		'defaults': {'cav_M_DM': 5e11, 'cav_R_halo': 200.0, 'cav_alpha': 2.5, 'cav_R_s': 0.05},
		'bounds': {'cav_M_DM': (1e10, 5e12), 'cav_R_halo': (50.0, 500.0),
				   'cav_alpha': (1.0, 4.0), 'cav_R_s': (0.01, 0.5)},
		'label': 'Cavity',
	},
	'cavity_wdm': {
		# Logistic cavity + WDM Burkert halo (right-handed neutrinos)
		'params': ['cav_M_DM', 'cav_R_halo', 'cav_alpha', 'cav_R_s', 'wdm_rho_0', 'wdm_r_0'],
		'defaults': {'cav_M_DM': 1e11, 'cav_R_halo': 200.0, 'cav_alpha': 2.5,
					 'cav_R_s': 0.05, 'wdm_rho_0': 1.0e7, 'wdm_r_0': 25.0},
		'bounds': {'cav_M_DM': (1e10, 5e12), 'cav_R_halo': (50.0, 500.0),
				   'cav_alpha': (1.0, 4.0), 'cav_R_s': (0.01, 0.5),
				   'wdm_rho_0': (1e4, 1e9), 'wdm_r_0': (1.0, 200.0)},
		'label': 'Cavity+WDM',
	},
	'cavity_rho': {
		# Logistic cavity parametrized by ρ_bg (physical) instead of M_DM
		# M_DM derived: M_DM = ρ_bg × 4π R_halo³ × ∫₀¹ s² deficit(s) ds
		'params': ['cav_rho_bg', 'cav_R_halo', 'cav_alpha', 'cav_R_s'],
		'defaults': {'cav_rho_bg': 3.9e4, 'cav_R_halo': 200.0, 'cav_alpha': 2.5, 'cav_R_s': 0.05},
		'bounds': {'cav_rho_bg': (1e3, 1e6), 'cav_R_halo': (50.0, 500.0),
				   'cav_alpha': (1.0, 4.0), 'cav_R_s': (0.01, 0.5)},
		'label': 'Cavity (ρ_bg)',
	},
	'cavity_wdm_rho': {
		# Logistic cavity (ρ_bg parametrized) + WDM Burkert halo
		'params': ['cav_rho_bg', 'cav_R_halo', 'cav_alpha', 'cav_R_s', 'wdm_rho_0', 'wdm_r_0'],
		'defaults': {'cav_rho_bg': 3.9e4, 'cav_R_halo': 200.0, 'cav_alpha': 2.5,
					 'cav_R_s': 0.05, 'wdm_rho_0': 1.0e7, 'wdm_r_0': 25.0},
		'bounds': {'cav_rho_bg': (1e3, 1e6), 'cav_R_halo': (50.0, 500.0),
				   'cav_alpha': (1.0, 4.0), 'cav_R_s': (0.01, 0.5),
				   'wdm_rho_0': (1e4, 1e9), 'wdm_r_0': (1.0, 200.0)},
		'label': 'Cavity+WDM (ρ_bg)',
	},
}

# Negative mass background density (cosmological reference)
RHO_BG = 3.9e4  # M☉/kpc³


class MWMassModel:
	"""
	Parametric mass model for the Milky Way.

	Components:
	  - SMBH: point mass (fixed)
	  - Bulge: Hernquist profile
	  - Stellar disk: exponential
	  - Gas disk: exponential (larger scale radius)
	  - DM halo: NFW or Burkert (selected via dm_type)
	"""

	def __init__(self, dm_type='nfw', M_bh=4.0e6, **params):
		self.dm_type = dm_type
		self.M_bh = M_bh

		dm_cfg = DM_CONFIGS[dm_type]
		self.PARAM_NAMES = BARYON_PARAM_NAMES + dm_cfg['params']
		self.DEFAULTS = {**BARYON_DEFAULTS, **dm_cfg['defaults']}
		self.BOUNDS = {**BARYON_BOUNDS, **dm_cfg['bounds']}
		self.dm_label = dm_cfg['label']

		self.params = {**self.DEFAULTS, **params}

	def M_enc_baryons(self, r):
		"""Total baryonic enclosed mass at radius r."""
		p = self.params
		return (M_enc_point(r, self.M_bh)
				+ M_enc_hernquist(r, p['M_bulge'], p['a_bulge'])
				+ M_enc_exponential_disk(r, p['M_disk'], p['r_disk'])
				+ M_enc_gas_disk(r, p['M_gas'], p['r_gas']))

	def M_enc_dm(self, r):
		"""DM halo enclosed mass at radius r."""
		p = self.params
		if self.dm_type == 'nfw':
			return M_enc_nfw(r, p['M_200'], p['c_200'])
		elif self.dm_type == 'burkert':
			return M_enc_burkert(r, p['rho_0'], p['r_0'])
		elif self.dm_type in ('cavity', 'cavity_rho'):
			return self._M_enc_cavity(r)
		elif self.dm_type in ('cavity_wdm', 'cavity_wdm_rho'):
			return self._M_enc_cavity(r) + M_enc_burkert(r, p['wdm_rho_0'], p['wdm_r_0'])
		else:
			raise ValueError(f"Unknown DM type: {self.dm_type}")

	def M_enc_total(self, r):
		"""Total enclosed mass (baryons + DM)."""
		return self.M_enc_baryons(r) + self.M_enc_dm(r)

	def v_circular_total(self, r):
		"""Total circular velocity [km/s]."""
		return v_circular(self.M_enc_total(r), r)

	def v_circular_components(self, r):
		"""Individual component circular velocities [km/s]."""
		p = self.params
		return {
			'SMBH':    v_circular(M_enc_point(r, self.M_bh), r),
			'Bulge':   v_circular(M_enc_hernquist(r, p['M_bulge'], p['a_bulge']), r),
			'Disk':    v_circular(M_enc_exponential_disk(r, p['M_disk'], p['r_disk']), r),
			'Gas':     v_circular(M_enc_gas_disk(r, p['M_gas'], p['r_gas']), r),
			'DM halo': v_circular(self.M_enc_dm(r), r),
			'Total':   self.v_circular_total(r),
		}

	def set_from_vector(self, x):
		"""Set parameters from a flat array (for optimizer)."""
		for i, name in enumerate(self.PARAM_NAMES):
			self.params[name] = x[i]

	def get_vector(self):
		"""Get parameters as a flat array."""
		return np.array([self.params[name] for name in self.PARAM_NAMES])

	def get_bounds(self):
		"""Get parameter bounds as list of (lo, hi) tuples."""
		return [self.BOUNDS[name] for name in self.PARAM_NAMES]

	def total_baryonic_mass(self):
		"""Sum of all baryonic component masses."""
		p = self.params
		return self.M_bh + p['M_bulge'] + p['M_disk'] + p['M_gas']

	def dm_virial_mass(self):
		"""Approximate virial mass of the DM halo."""
		if self.dm_type == 'nfw':
			return self.params['M_200']
		else:
			# For Burkert / cavity+WDM: use M_enc at 200 kpc as proxy
			return self.M_enc_dm(np.array([200.0]))[0]

	def _M_enc_cavity(self, r):
		"""
		Logistic cavity deficit M_enc(r).
		deficit(x) = R_s^α / (R_s^α + x^α), x = r/R_halo
		Supports two parametrizations:
		  - cav_M_DM: total DM mass given, ρ_bg derived
		  - cav_rho_bg: background density given, M_DM derived
		"""
		p = self.params
		R_halo = p['cav_R_halo']
		alpha = p['cav_alpha']
		R_s = p['cav_R_s']

		x = np.clip(np.asarray(r, dtype=float) / R_halo, 0.0, 1.0)

		# Tabulate cumulative deficit integral on fine grid
		s = np.linspace(0, 1, 500)
		Rs_a = R_s ** alpha
		deficit = Rs_a / (Rs_a + np.power(np.maximum(s, 1e-30), alpha))
		cum = np.concatenate([[0.0], integrate.cumulative_trapezoid(s**2 * deficit, s)])
		total = cum[-1]

		if total <= 0:
			return x * 0.0

		if 'cav_rho_bg' in p:
			M_DM = p['cav_rho_bg'] * 4 * np.pi * R_halo**3 * total
		else:
			M_DM = p['cav_M_DM']

		return M_DM * np.interp(x, s, cum / total)

	def _rho_cavity(self, r):
		"""Logistic cavity deficit density ρ(r) [M☉/kpc³]."""
		p = self.params
		R_halo = p['cav_R_halo']
		alpha = p['cav_alpha']
		R_s = p['cav_R_s']
		Rs_a = R_s ** alpha

		if 'cav_rho_bg' in p:
			rho_bg = p['cav_rho_bg']
		else:
			I, _ = integrate.quad(lambda s: s**2 * Rs_a / (Rs_a + s**alpha), 0, 1, limit=200)
			rho_bg = p['cav_M_DM'] / (4 * np.pi * R_halo**3 * I) if I > 0 else 0.0

		x = np.asarray(r, dtype=float) / R_halo
		deficit = Rs_a / (Rs_a + np.power(np.maximum(x, 1e-30), alpha))
		return np.where(np.asarray(r, dtype=float) <= R_halo, rho_bg * deficit, 0.0)

	def cav_rho_bg(self):
		"""ρ_bg: direct if parametrized, otherwise derived from M_DM."""
		p = self.params
		if 'cav_rho_bg' in p:
			return p['cav_rho_bg']
		Rs_a = p['cav_R_s'] ** p['cav_alpha']
		I, _ = integrate.quad(
			lambda s: s**2 * Rs_a / (Rs_a + s**p['cav_alpha']), 0, 1, limit=200)
		return p['cav_M_DM'] / (4 * np.pi * p['cav_R_halo']**3 * I) if I > 0 else 0.0

	def cav_M_dm(self):
		"""Total cavity DM mass: direct if parametrized, otherwise derived from ρ_bg."""
		p = self.params
		if 'cav_M_DM' in p:
			return p['cav_M_DM']
		Rs_a = p['cav_R_s'] ** p['cav_alpha']
		I, _ = integrate.quad(
			lambda s: s**2 * Rs_a / (Rs_a + s**p['cav_alpha']), 0, 1, limit=200)
		return p['cav_rho_bg'] * 4 * np.pi * p['cav_R_halo']**3 * I if I > 0 else 0.0

	def dm_density(self, r):
		"""DM density profile ρ(r) [M☉/kpc³]."""
		p = self.params
		if self.dm_type == 'nfw':
			rho_crit = 126.0
			r_200 = (3 * p['M_200'] / (4 * np.pi * 200 * rho_crit))**(1.0/3.0)
			r_s = r_200 / p['c_200']
			rho_s = p['M_200'] / (4 * np.pi * r_s**3 * (
				np.log(1 + p['c_200']) - p['c_200'] / (1 + p['c_200'])))
			x = r / r_s
			return rho_s / (x * (1 + x)**2)
		elif self.dm_type == 'burkert':
			x = r / p['r_0']
			return p['rho_0'] / ((1 + x) * (1 + x**2))
		elif self.dm_type in ('cavity', 'cavity_rho', 'cavity_wdm', 'cavity_wdm_rho'):
			rho_cav = self._rho_cavity(r)
			if self.dm_type in ('cavity_wdm', 'cavity_wdm_rho'):
				x = r / p['wdm_r_0']
				rho_wdm = p['wdm_rho_0'] / ((1 + x) * (1 + x**2))
				return rho_cav + rho_wdm
			return rho_cav


# ─────────────────────────────────────────────────────────────────────────────
# Fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_rotation_curve(data, model, free_params=None):
	"""
	Fit the mass model to observed rotation curve data.

	Parameters
	----------
	data : dict with 'r', 'v', 'v_err'
	model : MWMassModel instance
	free_params : list of parameter names to fit (None = all)

	Returns
	-------
	model : updated MWMassModel with best-fit parameters
	result : scipy OptimizeResult
	"""
	if free_params is None:
		free_params = model.PARAM_NAMES

	# Map free params to indices
	idx = [model.PARAM_NAMES.index(p) for p in free_params]

	x0_full = model.get_vector()
	bounds_full = model.get_bounds()

	x0 = x0_full[idx]
	bounds = [bounds_full[i] for i in idx]

	# Work in log-space for masses (better conditioning)
	# Identify which params are masses (large values) vs scale params
	is_mass = np.array([model.DEFAULTS[free_params[j]] > 1e6 for j in range(len(free_params))])

	def pack(x):
		"""Transform to optimizer space."""
		xp = np.copy(x)
		xp[is_mass] = np.log10(x[is_mass])
		return xp

	def unpack(xp):
		"""Transform back from optimizer space."""
		x = np.copy(xp)
		x[is_mass] = 10**xp[is_mass]
		return x

	def cost(xp):
		"""Chi-squared cost function."""
		x = unpack(xp)
		# Set free params
		for j, pidx in enumerate(idx):
			x0_full[pidx] = x[j]
		model.set_from_vector(x0_full)

		v_model = model.v_circular_total(data['r'])
		chi2 = np.sum(((data['v'] - v_model) / data['v_err'])**2)
		return chi2

	# Transform bounds
	bounds_packed = []
	for j, (lo, hi) in enumerate(bounds):
		if is_mass[j]:
			bounds_packed.append((np.log10(lo), np.log10(hi)))
		else:
			bounds_packed.append((lo, hi))

	xp0 = pack(x0)

	result = minimize(cost, xp0, method='L-BFGS-B', bounds=bounds_packed,
					  options={'maxiter': 5000, 'ftol': 1e-12})

	# Unpack best-fit
	x_best = unpack(result.x)
	for j, pidx in enumerate(idx):
		x0_full[pidx] = x_best[j]
	model.set_from_vector(x0_full)

	# Compute reduced chi-squared
	n_data = len(data['r'])
	n_free = len(free_params)
	chi2 = result.fun
	chi2_red = chi2 / max(n_data - n_free, 1)

	print(f"\n=== Fit Results ===")
	print(f"  χ² = {chi2:.1f},  χ²_red = {chi2_red:.2f}  (N_data={n_data}, N_free={n_free})")
	print(f"\n  Best-fit parameters:")
	for name in model.PARAM_NAMES:
		val = model.params[name]
		if val > 1e6:
			print(f"    {name:10s} = {val:.3e} M☉")
		else:
			print(f"    {name:10s} = {val:.3f} kpc")
	M_dm_vir = model.dm_virial_mass()
	print(f"\n  Total baryonic mass: {model.total_baryonic_mass():.3e} M☉")
	print(f"  DM halo mass:        {M_dm_vir:.3e} M☉  ({model.dm_label})")
	print(f"  Baryon fraction:     {model.total_baryonic_mass() / M_dm_vir:.3f}")

	return model, result


# ─────────────────────────────────────────────────────────────────────────────
# Infer DM profile from data (model-independent)
# ─────────────────────────────────────────────────────────────────────────────

def infer_dm_profile(data, model):
	"""
	Given observed v_c and a baryonic model, infer the required DM enclosed mass
	at each data radius (model-independent).

	M_total(r) = v_c² × r / G
	M_DM(r) = M_total(r) - M_baryons(r)

	Returns dict with 'r', 'M_total', 'M_bary', 'M_dm', 'rho_dm'.
	"""
	G = G_kpc_Msun_Gyr
	r = data['r']
	v = data['v']  # km/s
	v_err = data['v_err']

	# Convert v_c to kpc/Gyr
	v_sim = v / kpc_per_Gyr_to_kms

	# Total enclosed mass from observed v_c
	M_total = v_sim**2 * r / G
	M_total_lo = ((v - v_err) / kpc_per_Gyr_to_kms)**2 * r / G
	M_total_hi = ((v + v_err) / kpc_per_Gyr_to_kms)**2 * r / G

	# Baryonic enclosed mass from the model
	M_bary = model.M_enc_baryons(r)

	# Required DM
	M_dm = M_total - M_bary
	M_dm_lo = M_total_lo - M_bary
	M_dm_hi = M_total_hi - M_bary

	# Estimate local DM density from finite differences
	# ρ_DM(r) ≈ (1/4πr²) × dM_DM/dr
	rho_dm = np.zeros_like(r)
	for i in range(1, len(r) - 1):
		dr = r[i+1] - r[i-1]
		dM = M_dm[i+1] - M_dm[i-1]
		rho_dm[i] = dM / (4 * np.pi * r[i]**2 * dr)

	return {
		'r': r,
		'M_total': M_total,
		'M_total_lo': M_total_lo,
		'M_total_hi': M_total_hi,
		'M_bary': M_bary,
		'M_dm': M_dm,
		'M_dm_lo': M_dm_lo,
		'M_dm_hi': M_dm_hi,
		'rho_dm': rho_dm,
	}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_decomposition(data, model, dm_profile, save_path='mw_rotation_decomposition.png',
					   model_alt=None, dm_profile_alt=None):
	"""
	Plot the full rotation curve decomposition with 6 panels.
	Optionally overlay an alternative DM model for comparison.
	"""
	r_plot = np.linspace(0.1, 110, 500)
	comps = model.v_circular_components(r_plot)
	if model_alt is not None:
		comps_alt = model_alt.v_circular_components(r_plot)

	fig, axes = plt.subplots(2, 3, figsize=(18, 11))
	fig.patch.set_facecolor('#111111')
	fig.suptitle('Milky Way Rotation Curve Decomposition', fontsize=16, fontweight='bold', color='white')
	for ax in axes.ravel():
		ax.set_facecolor('#0d0d0d')
		ax.tick_params(colors='#aaaaaa')
		ax.xaxis.label.set_color('#cccccc')
		ax.yaxis.label.set_color('#cccccc')
		ax.title.set_color('#ffffff')
		for spine in ax.spines.values():
			spine.set_edgecolor('#333333')

	# Color scheme
	colors = {
		'SMBH': '#aaaaaa', 'Bulge': 'orange', 'Disk': '#44cc44',
		'Gas': 'cyan', 'DM halo': '#4488ff', 'Total': '#ff4444',
	}

	# Data source colors and markers
	src_colors = {'Sofue': '#aaaaaa', 'Eilers2019': '#ffffff', 'Huang2016': '#99aabb'}
	src_markers = {'Sofue': 's', 'Eilers2019': 'o', 'Huang2016': 'D'}
	# ─── Panel 1: Rotation curve decomposition ──────────────────────────────────────
	ax = axes[0, 0]
	sources = set(data['source'])
	for src in sources:
		mask = np.array([s == src for s in data['source']])
		ax.errorbar(data['r'][mask], data['v'][mask], yerr=data['v_err'][mask],
					fmt=src_markers.get(src, 'o'), ms=4, capsize=2, alpha=0.7,
					color=src_colors.get(src, '#aaaaaa'), label=f'Data ({src})')

	# Primary model component decomposition only (no alt overlay — too crowded)
	for name in ['SMBH', 'Bulge', 'Disk', 'Gas', 'DM halo']:
		ax.plot(r_plot, comps[name], '--', color=colors[name], lw=1.5, alpha=0.7, label=name)
	ax.plot(r_plot, comps['Total'], '-', color=colors['Total'], lw=2.5, label='Total')

	ax.set_xlabel('r [kpc]')
	ax.set_ylabel('v_c [km/s]')
	ax.set_title(f'Rotation Curve Decomposition ({model.dm_label})')
	ax.legend(fontsize=7, ncol=2, facecolor='#222222', labelcolor='white', edgecolor='#555555')
	ax.set_xlim(0, 30)
	ax.set_ylim(0, 300)

	# ─── Panel 2: Wide view (full range) ─────────────────────────────────
	ax = axes[0, 1]
	for src in sources:
		mask = np.array([s == src for s in data['source']])
		ax.errorbar(data['r'][mask], data['v'][mask], yerr=data['v_err'][mask],
					fmt=src_markers.get(src, 'o'), ms=4, capsize=2, alpha=0.7,
					color=src_colors.get(src, '#aaaaaa'), label=f'Data ({src})')
	ax.plot(r_plot, comps['Total'], '-', color='red', lw=2.5, label=f'Total ({model.dm_label})')
	ax.plot(r_plot, comps['DM halo'], '--', color='blue', lw=1.5, alpha=0.7,
		   label=f'DM ({model.dm_label})')
	if model_alt is not None:
		ax.plot(r_plot, comps_alt['Total'], '-', color='purple', lw=2,
			   label=f'Total ({model_alt.dm_label})')
		ax.plot(r_plot, comps_alt['DM halo'], ':', color='purple', lw=1.5, alpha=0.7,
			   label=f'DM ({model_alt.dm_label})')

	# Baryons only — show for both models
	v_bary = v_circular(model.M_enc_baryons(r_plot), r_plot)
	ax.plot(r_plot, v_bary, '--', color='green', lw=1.5, alpha=0.7,
		   label=f'Baryons ({model.dm_label})')
	if model_alt is not None:
		v_bary_alt = v_circular(model_alt.M_enc_baryons(r_plot), r_plot)
		ax.plot(r_plot, v_bary_alt, ':', color='green', lw=1.5, alpha=0.7,
			   label=f'Baryons ({model_alt.dm_label})')

	ax.set_xlabel('r [kpc]')
	ax.set_ylabel('v_c [km/s]')
	ax.set_title('Full Range Rotation Curve')
	ax.legend(fontsize=7, facecolor='#222222', labelcolor='white', edgecolor='#555555')
	ax.set_xlim(0, 110)
	ax.set_ylim(0, 300)

	# ─── Panel 3: Residuals ──────────────────────────────────────────────
	ax = axes[0, 2]
	v_model_at_data = model.v_circular_total(data['r'])
	residuals = data['v'] - v_model_at_data
	residuals_sigma = residuals / data['v_err']

	for src in sources:
		mask = np.array([s == src for s in data['source']])
		ax.errorbar(data['r'][mask], residuals[mask], yerr=data['v_err'][mask],
					fmt=src_markers.get(src, 'o'), ms=4, capsize=2, alpha=0.7,
					color=src_colors.get(src, '#aaaaaa'), label=f'{src}')
	ax.axhline(0, color='red', ls='-', alpha=0.5)
	ax.fill_between([0, 110], [-5, -5], [5, 5], alpha=0.1, color='red')
	ax.set_xlabel('r [kpc]')
	ax.set_ylabel('v_obs − v_model [km/s]')
	ax.set_title('Residuals')
	ax.legend(fontsize=7, facecolor='#222222', labelcolor='white', edgecolor='#555555')
	ax.set_xlim(0, 110)

	# ─── Panel 4: Enclosed mass ──────────────────────────────────────────
	ax = axes[1, 0]
	ax.plot(r_plot, model.M_enc_total(r_plot), 'r-', lw=2.5, label='Total')
	ax.plot(r_plot, model.M_enc_dm(r_plot), 'b--', lw=1.5, label='DM halo')
	ax.plot(r_plot, model.M_enc_baryons(r_plot), 'g--', lw=1.5, label='Baryons')

	# Inferred total from data
	ax.fill_between(dm_profile['r'], dm_profile['M_total_lo'], dm_profile['M_total_hi'],
					alpha=0.2, color='#888888')
	ax.scatter(dm_profile['r'], dm_profile['M_total'], c='#888888', s=10, zorder=5,
			   label='Observed M_total')

	ax.set_xlabel('r [kpc]')
	ax.set_ylabel('M_enc [M☉]')
	ax.set_title('Enclosed Mass Profiles')
	ax.legend(fontsize=7, loc='lower right', facecolor='#222222', labelcolor='white', edgecolor='#555555')
	ax.set_yscale('log')
	ax.set_xlim(0, 110)
	ax.set_ylim(1e8, 2e12)

	# ─── Panel 5: Required DM mass ──────────────────────────────────────
	ax = axes[1, 1]

	# Required DM from primary model's baryonic fit
	r_dm = dm_profile['r']
	mask_pos = dm_profile['M_dm'] > 0
	ax.fill_between(r_dm[mask_pos], dm_profile['M_dm_lo'][mask_pos],
					dm_profile['M_dm_hi'][mask_pos], alpha=0.2, color='red')
	ax.scatter(r_dm[mask_pos], dm_profile['M_dm'][mask_pos],
			   c='red', s=10, zorder=5, label=f'Required ({model.dm_label} baryons)')

	# Required DM from alt model's baryonic fit
	if dm_profile_alt is not None:
		r_dm_a = dm_profile_alt['r']
		mask_a = dm_profile_alt['M_dm'] > 0
		ax.fill_between(r_dm_a[mask_a], dm_profile_alt['M_dm_lo'][mask_a],
						dm_profile_alt['M_dm_hi'][mask_a], alpha=0.2, color='purple')
		ax.scatter(r_dm_a[mask_a], dm_profile_alt['M_dm'][mask_a],
				   c='purple', s=10, zorder=5, label=f'Required ({model_alt.dm_label} baryons)')

	# Model DM enclosed mass curves
	ax.plot(r_plot, model.M_enc_dm(r_plot), 'r-', lw=2, label=f'{model.dm_label} fit')
	if model_alt is not None:
		ax.plot(r_plot, model_alt.M_enc_dm(r_plot), 'purple', ls='-', lw=2,
			   label=f'{model_alt.dm_label} fit')

	ax.set_xlabel('r [kpc]')
	ax.set_ylabel('M_DM(r) [M☉]')
	ax.set_title('Dark Matter Enclosed Mass')
	ax.legend(fontsize=7, loc='lower right', facecolor='#222222', labelcolor='white', edgecolor='#555555')
	ax.set_yscale('log')
	ax.set_xlim(0, 110)
	ax.set_ylim(1e8, 2e12)

	# ─── Panel 6: DM density profile ────────────────────────────────────
	ax = axes[1, 2]
	# Primary model density
	rho_primary = model.dm_density(r_plot)
	ax.plot(r_plot, rho_primary, 'b-', lw=2, label=f'{model.dm_label} fit')

	# Alternative model density
	if model_alt is not None:
		rho_alt = model_alt.dm_density(r_plot)
		ax.plot(r_plot, rho_alt, 'purple', ls='-', lw=2, label=f'{model_alt.dm_label} fit')

	# Inferred from data (primary model's baryonic subtraction)
	valid = dm_profile['rho_dm'] > 0
	ax.scatter(dm_profile['r'][valid], dm_profile['rho_dm'][valid],
			   c='red', s=12, alpha=0.7, zorder=5,
			   label=f'Inferred ({model.dm_label} baryons)')

	# Inferred from alt model's baryonic subtraction
	if dm_profile_alt is not None:
		valid_a = dm_profile_alt['rho_dm'] > 0
		ax.scatter(dm_profile_alt['r'][valid_a], dm_profile_alt['rho_dm'][valid_a],
				   c='purple', s=12, alpha=0.7, zorder=5,
				   label=f'Inferred ({model_alt.dm_label} baryons)')

	# Cavity background density for comparison
	ax.axhline(RHO_BG, color='green', ls=':', alpha=0.7, label=f'ρ_bg = {RHO_BG:.0e} M☉/kpc³')

	# Local DM density at R☉
	R_sun = 8.122
	rho_local = model.dm_density(np.array([R_sun]))[0]
	ax.axvline(R_sun, color='orange', ls='--', alpha=0.5)
	ax.annotate(f'ρ_DM(R☉)={rho_local:.2e}\nM☉/kpc³',
				xy=(R_sun, rho_local), fontsize=7, ha='left', color='#cccccc')
	if model_alt is not None:
		rho_local_alt = model_alt.dm_density(np.array([R_sun]))[0]
		ax.annotate(f'{model_alt.dm_label}: {rho_local_alt:.2e}',
					xy=(R_sun * 1.2, rho_local_alt * 0.5), fontsize=7, color='#aa88ff')

	ax.set_xlabel('r [kpc]')
	ax.set_ylabel('ρ_DM [M☉/kpc³]')
	ax.set_title('DM Density Profile')
	ax.legend(fontsize=7, facecolor='#222222', labelcolor='white', edgecolor='#555555')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlim(0.3, 300)
	ax.set_ylim(1e2, 1e9)

	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#111111')
	plt.show()
	print(f"\nPlot saved to {save_path}")


def plot_cavity_wdm(data, model_cwdm, dm_profile_cwdm,
					model_nfw=None, model_burkert=None,
					save_path='mw_cavity_wdm_decomposition.png'):
	"""
	Dedicated plot for the Cavity + WDM composite model.
	6 panels focused on showing the two-component DM structure.
	"""
	r_plot = np.linspace(0.1, 110, 500)
	comps = model_cwdm.v_circular_components(r_plot)

	# Cavity and WDM contributions separately
	p = model_cwdm.params
	M_cavity = model_cwdm._M_enc_cavity(r_plot)
	M_wdm = M_enc_burkert(r_plot, p['wdm_rho_0'], p['wdm_r_0'])
	v_cavity = v_circular(M_cavity, r_plot)
	v_wdm = v_circular(M_wdm, r_plot)

	R_halo = p['cav_R_halo']
	rho_bg = model_cwdm.cav_rho_bg()

	fig, axes = plt.subplots(2, 3, figsize=(18, 11))
	fig.patch.set_facecolor('#111111')
	fig.suptitle('Cavity + Warm Dark Matter (Right-Handed Neutrino) Model',
				 fontsize=16, fontweight='bold', color='white')
	for ax in axes.ravel():
		ax.set_facecolor('#0d0d0d')
		ax.tick_params(colors='#aaaaaa')
		ax.xaxis.label.set_color('#cccccc')
		ax.yaxis.label.set_color('#cccccc')
		ax.title.set_color('#ffffff')
		for spine in ax.spines.values():
			spine.set_edgecolor('#333333')

	# Data styling
	sources = set(data['source'])
	src_colors = {'Sofue': '#aaaaaa', 'Eilers2019': '#ffffff', 'Huang2016': '#99aabb'}
	src_markers = {'Sofue': 's', 'Eilers2019': 'o', 'Huang2016': 'D'}

	colors = {
		'SMBH': '#aaaaaa', 'Bulge': 'orange', 'Disk': '#44cc44',
		'Gas': 'cyan', 'DM halo': '#4488ff', 'Total': '#ff4444',
	}

	# ─── Panel 1: Full decomposition ─────────────────────────────────────
	ax = axes[0, 0]
	for src in sources:
		mask = np.array([s == src for s in data['source']])
		ax.errorbar(data['r'][mask], data['v'][mask], yerr=data['v_err'][mask],
					fmt=src_markers.get(src, 'o'), ms=4, capsize=2, alpha=0.7,
					color=src_colors.get(src, '#aaaaaa'), label=f'Data ({src})')

	for name in ['SMBH', 'Bulge', 'Disk', 'Gas']:
		ax.plot(r_plot, comps[name], '--', color=colors[name], lw=1.5, alpha=0.7, label=name)
	ax.plot(r_plot, v_cavity, '--', color='forestgreen', lw=1.5, alpha=0.8, label='Cavity')
	ax.plot(r_plot, v_wdm, '--', color='blueviolet', lw=1.5, alpha=0.8, label='WDM halo')
	ax.plot(r_plot, comps['Total'], '-', color='red', lw=2.5, label='Total')

	ax.set_xlabel('r [kpc]')
	ax.set_ylabel('v_c [km/s]')
	ax.set_title('Rotation Curve Decomposition (Cavity+WDM)')
	ax.legend(fontsize=7, ncol=2, facecolor='#222222', labelcolor='white', edgecolor='#555555')
	ax.set_xlim(0, 30)
	ax.set_ylim(0, 300)

	# ─── Panel 2: Three-model comparison (full range) ────────────────────
	ax = axes[0, 1]
	for src in sources:
		mask = np.array([s == src for s in data['source']])
		ax.errorbar(data['r'][mask], data['v'][mask], yerr=data['v_err'][mask],
					fmt=src_markers.get(src, 'o'), ms=4, capsize=2, alpha=0.7,
					color=src_colors.get(src, '#aaaaaa'), label=f'Data ({src})')
	ax.plot(r_plot, comps['Total'], '-', color='red', lw=2.5, label='Cavity+WDM')
	if model_nfw is not None:
		ax.plot(r_plot, model_nfw.v_circular_total(r_plot), '-', color='blue', lw=1.5,
				alpha=0.7, label='NFW')
	if model_burkert is not None:
		ax.plot(r_plot, model_burkert.v_circular_total(r_plot), '-', color='purple', lw=1.5,
				alpha=0.7, label='Burkert')

	ax.set_xlabel('r [kpc]')
	ax.set_ylabel('v_c [km/s]')
	ax.set_title('Three-Model Comparison')
	ax.legend(fontsize=7, facecolor='#222222', labelcolor='white', edgecolor='#555555')
	ax.set_xlim(0, 110)
	ax.set_ylim(0, 300)

	# ─── Panel 3: Residuals ──────────────────────────────────────────────
	ax = axes[0, 2]
	v_model_at_data = model_cwdm.v_circular_total(data['r'])
	residuals = data['v'] - v_model_at_data
	for src in sources:
		mask = np.array([s == src for s in data['source']])
		ax.errorbar(data['r'][mask], residuals[mask], yerr=data['v_err'][mask],
					fmt=src_markers.get(src, 'o'), ms=4, capsize=2, alpha=0.7,
					color=src_colors.get(src, '#aaaaaa'), label=f'{src}')
	ax.axhline(0, color='red', ls='-', alpha=0.5)
	ax.fill_between([0, 110], [-5, -5], [5, 5], alpha=0.1, color='red')
	ax.set_xlabel('r [kpc]')
	ax.set_ylabel('v_obs − v_model [km/s]')
	ax.set_title('Residuals (Cavity+WDM)')
	ax.legend(fontsize=7, facecolor='#222222', labelcolor='white', edgecolor='#555555')
	ax.set_xlim(0, 110)

	# ─── Panel 4: DM enclosed mass — cavity vs WDM breakdown ────────────
	ax = axes[1, 0]
	ax.plot(r_plot, model_cwdm.M_enc_dm(r_plot), 'r-', lw=2.5, label='Total DM (Cavity+WDM)')
	ax.plot(r_plot, M_cavity, '--', color='forestgreen', lw=2,
			label=f'Cavity (R_h={R_halo:.0f}, α={p["cav_alpha"]:.1f}, Rs={p["cav_R_s"]:.2f})')
	ax.plot(r_plot, M_wdm, '--', color='blueviolet', lw=2, label='WDM halo')
	if model_nfw is not None:
		ax.plot(r_plot, model_nfw.M_enc_dm(r_plot), 'b:', lw=1.5, alpha=0.7, label='NFW')
	if model_burkert is not None:
		ax.plot(r_plot, model_burkert.M_enc_dm(r_plot), 'purple', ls=':', lw=1.5,
				alpha=0.7, label='Burkert')

	# Data-inferred required DM
	r_dm = dm_profile_cwdm['r']
	mask_pos = dm_profile_cwdm['M_dm'] > 0
	ax.scatter(r_dm[mask_pos], dm_profile_cwdm['M_dm'][mask_pos],
			   c='#888888', s=10, zorder=5, alpha=0.7, label='Required DM (data)')

	ax.set_xlabel('r [kpc]')
	ax.set_ylabel('M_DM(r) [M☉]')
	ax.set_title('DM Enclosed Mass Breakdown')
	ax.legend(fontsize=7, facecolor='#222222', labelcolor='white', edgecolor='#555555')
	ax.set_yscale('log')
	ax.set_xlim(0, 110)
	ax.set_ylim(1e8, 2e12)

	# ─── Panel 5: DM density — cavity logistic + WDM core ──────────────
	ax = axes[1, 1]
	rho_total = model_cwdm.dm_density(r_plot)
	rho_cavity = model_cwdm._rho_cavity(r_plot)
	x_wdm = r_plot / p['wdm_r_0']
	rho_wdm = p['wdm_rho_0'] / ((1 + x_wdm) * (1 + x_wdm**2))

	ax.plot(r_plot, rho_total, 'r-', lw=2.5, label='Total (Cavity+WDM)')
	ax.fill_between(r_plot, 0, np.maximum(rho_cavity, 1), alpha=0.15, color='forestgreen')
	ax.plot(r_plot, np.maximum(rho_cavity, 1), '--', color='forestgreen', lw=1.5,
			label=f'Cavity (ρ_bg={rho_bg:.1e})')
	ax.plot(r_plot, rho_wdm, '--', color='blueviolet', lw=1.5, label='WDM halo')

	if model_nfw is not None:
		ax.plot(r_plot, model_nfw.dm_density(r_plot), 'b:', lw=1.5, alpha=0.7, label='NFW')
	if model_burkert is not None:
		ax.plot(r_plot, model_burkert.dm_density(r_plot), 'purple', ls=':', lw=1.5,
				alpha=0.7, label='Burkert')

	# R☉ marker
	R_sun = 8.122
	rho_local = model_cwdm.dm_density(np.array([R_sun]))[0]
	ax.axvline(R_sun, color='orange', ls='--', alpha=0.5)
	msun_kpc3_to_gev_cm3 = 2.63e-8
	ax.annotate(f'ρ(R☉)={rho_local*msun_kpc3_to_gev_cm3:.2f} GeV/cm³',
				xy=(R_sun, rho_local), fontsize=7, ha='left', color='#cccccc')

	ax.set_xlabel('r [kpc]')
	ax.set_ylabel('ρ_DM [M☉/kpc³]')
	ax.set_title('DM Density Profile')
	ax.legend(fontsize=7, facecolor='#222222', labelcolor='white', edgecolor='#555555')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlim(0.3, 300)
	ax.set_ylim(1e2, 1e9)

	# ─── Panel 6: Cavity fraction of total DM ───────────────────────────
	ax = axes[1, 2]
	M_dm_total = model_cwdm.M_enc_dm(r_plot)
	cavity_frac = np.where(M_dm_total > 0, M_cavity / M_dm_total * 100, 0)
	wdm_frac = np.where(M_dm_total > 0, M_wdm / M_dm_total * 100, 0)

	ax.fill_between(r_plot, 0, cavity_frac, alpha=0.3, color='forestgreen', label='Cavity')
	ax.fill_between(r_plot, cavity_frac, 100, alpha=0.3, color='blueviolet', label='WDM halo')
	ax.plot(r_plot, cavity_frac, '-', color='forestgreen', lw=2)
	ax.axhline(50, color='#888888', ls=':', alpha=0.6)
	ax.axvline(R_halo, color='forestgreen', ls='--', alpha=0.5, lw=1)
	ax.annotate(f'R_halo={R_halo:.0f} kpc', xy=(R_halo, 85), fontsize=8, color='#88cc88')

	# Mark where cavity = WDM
	cross_idx = np.argmin(np.abs(cavity_frac - 50))
	if cavity_frac[0] > 50:
		ax.annotate(f'Cavity=WDM at {r_plot[cross_idx]:.0f} kpc',
					xy=(r_plot[cross_idx], 50), fontsize=8, color='#cccccc',
					xytext=(r_plot[cross_idx]+5, 60),
					arrowprops=dict(arrowstyle='->', color='#888888'))

	ax.set_xlabel('r [kpc]')
	ax.set_ylabel('Fraction of total DM [%]')
	ax.set_title('Cavity vs WDM Contribution')
	ax.legend(fontsize=8, facecolor='#222222', labelcolor='white', edgecolor='#555555')
	ax.set_xlim(0, 110)
	ax.set_ylim(0, 100)

	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#111111')
	plt.show()
	print(f"\nPlot saved to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
	# Load data
	data = load_mw_rotation_curve()
	print(f"Loaded {len(data['r'])} data points from {len(set(data['source']))} sources")
	print(f"  Range: {data['r'].min():.1f} - {data['r'].max():.1f} kpc")
	print(f"  v_c range: {data['v'].min():.0f} - {data['v'].max():.0f} km/s")

	# ─── NFW fit ─────────────────────────────────────────────────────────
	print("\n" + "="*70)
	print("  NFW DM HALO FIT")
	print("="*70)
	model_nfw = MWMassModel(dm_type='nfw')
	model_nfw, result_nfw = fit_rotation_curve(data, model_nfw)
	dm_profile_nfw = infer_dm_profile(data, model_nfw)

	# ─── Burkert fit ─────────────────────────────────────────────────────
	print("\n" + "="*70)
	print("  BURKERT (CORED) DM HALO FIT")
	print("="*70)
	model_burkert = MWMassModel(dm_type='burkert')
	model_burkert, result_burkert = fit_rotation_curve(data, model_burkert)
	dm_profile_burkert = infer_dm_profile(data, model_burkert)

	# ─── Cavity (logistic) fit ───────────────────────────────────────────
	print("\n" + "="*70)
	print("  CAVITY (LOGISTIC DEFICIT PROFILE) FIT")
	print("="*70)
	model_cavity = MWMassModel(dm_type='cavity')
	model_cavity, result_cavity = fit_rotation_curve(data, model_cavity)
	dm_profile_cavity = infer_dm_profile(data, model_cavity)

	p_cav = model_cavity.params
	rho_bg_cav = model_cavity.cav_rho_bg()
	print(f"\n  Cavity profile details:")
	print(f"    M_DM         = {p_cav['cav_M_DM']:.3e} M☉")
	print(f"    R_halo       = {p_cav['cav_R_halo']:.1f} kpc")
	print(f"    α            = {p_cav['cav_alpha']:.3f}")
	print(f"    R_s          = {p_cav['cav_R_s']:.4f}")
	print(f"    ρ_bg (derived) = {rho_bg_cav:.1f} M☉/kpc³")
	print(f"    M_DM / M_bary  = {p_cav['cav_M_DM'] / model_cavity.total_baryonic_mass():.1f}")

	# ─── Cavity + WDM fit ────────────────────────────────────────────────
	print("\n" + "="*70)
	print("  CAVITY + WARM DARK MATTER (RIGHT-HANDED NEUTRINOS) FIT")
	print("="*70)
	model_cwdm = MWMassModel(dm_type='cavity_wdm')
	model_cwdm, result_cwdm = fit_rotation_curve(data, model_cwdm)
	dm_profile_cwdm = infer_dm_profile(data, model_cwdm)

	# Print cavity+WDM breakdown
	p_cwdm = model_cwdm.params
	rho_bg_cwdm = model_cwdm.cav_rho_bg()
	r_check = np.array([5, 8.122, 10, 25, 50, 100.0])
	M_cavity_at_r = model_cwdm._M_enc_cavity(r_check)
	M_wdm_at_r = M_enc_burkert(r_check, p_cwdm['wdm_rho_0'], p_cwdm['wdm_r_0'])
	M_total_dm_at_r = model_cwdm.M_enc_dm(r_check)

	print(f"\n  Cavity parameters (fitted logistic profile):")
	print(f"    M_DM         = {p_cwdm['cav_M_DM']:.3e} M☉")
	print(f"    R_halo       = {p_cwdm['cav_R_halo']:.1f} kpc")
	print(f"    α            = {p_cwdm['cav_alpha']:.3f}")
	print(f"    R_s          = {p_cwdm['cav_R_s']:.4f}")
	print(f"    ρ_bg (derived) = {rho_bg_cwdm:.1f} M☉/kpc³")
	print(f"\n  WDM halo parameters (fitted):")
	print(f"    ρ_0 (WDM)    = {p_cwdm['wdm_rho_0']:.3e} M☉/kpc³")
	print(f"    r_0 (WDM)    = {p_cwdm['wdm_r_0']:.3f} kpc")
	print(f"\n  DM mass breakdown by radius [M☉]:")
	print(f"  {'r [kpc]':>10s}  {'Cavity':>12s}  {'WDM':>12s}  {'Total DM':>12s}  {'Cavity %':>10s}")
	for i, rv in enumerate(r_check):
		cav_pct = M_cavity_at_r[i] / M_total_dm_at_r[i] * 100 if M_total_dm_at_r[i] > 0 else 0
		print(f"  {rv:10.1f}  {M_cavity_at_r[i]:12.3e}  {M_wdm_at_r[i]:12.3e}  {M_total_dm_at_r[i]:12.3e}  {cav_pct:9.1f}%")

	# ─── Cavity + WDM (ρ_bg parametrized, free fit) ─────────────────────
	print("\n" + "="*70)
	print("  CAVITY + WDM (ρ_bg PARAMETRIZED, FREE FIT)")
	print("="*70)
	model_cwdm_rho = MWMassModel(dm_type='cavity_wdm_rho')
	model_cwdm_rho, result_cwdm_rho = fit_rotation_curve(data, model_cwdm_rho)
	dm_profile_cwdm_rho = infer_dm_profile(data, model_cwdm_rho)

	p_rho_free = model_cwdm_rho.params
	cav_M_dm_free = model_cwdm_rho.cav_M_dm()
	print(f"\n  Cavity (ρ_bg parametrized, fitted):")
	print(f"    ρ_bg (fitted) = {p_rho_free['cav_rho_bg']:.1f} M☉/kpc³")
	print(f"    R_halo        = {p_rho_free['cav_R_halo']:.1f} kpc")
	print(f"    α             = {p_rho_free['cav_alpha']:.3f}")
	print(f"    R_s           = {p_rho_free['cav_R_s']:.4f}")
	print(f"    M_DM (derived)= {cav_M_dm_free:.3e} M☉")
	print(f"    ρ_bg / ρ_cosmo = {p_rho_free['cav_rho_bg'] / RHO_BG:.1f}×")

	# ─── Cavity + WDM (ρ_bg fixed at cosmological value) ────────────────
	print("\n" + "="*70)
	print(f"  CAVITY + WDM (ρ_bg FIXED AT {RHO_BG:.0f} M☉/kpc³)")
	print("="*70)
	model_cwdm_fixed = MWMassModel(dm_type='cavity_wdm_rho')
	free_fixed = [p for p in model_cwdm_fixed.PARAM_NAMES if p != 'cav_rho_bg']
	model_cwdm_fixed, result_cwdm_fixed = fit_rotation_curve(
		data, model_cwdm_fixed, free_params=free_fixed)
	dm_profile_cwdm_fixed = infer_dm_profile(data, model_cwdm_fixed)

	p_rho_fix = model_cwdm_fixed.params
	cav_M_dm_fixed = model_cwdm_fixed.cav_M_dm()
	print(f"\n  Cavity (ρ_bg fixed at cosmological value):")
	print(f"    ρ_bg (fixed)  = {p_rho_fix['cav_rho_bg']:.1f} M☉/kpc³")
	print(f"    R_halo        = {p_rho_fix['cav_R_halo']:.1f} kpc")
	print(f"    α             = {p_rho_fix['cav_alpha']:.3f}")
	print(f"    R_s           = {p_rho_fix['cav_R_s']:.4f}")
	print(f"    M_DM (derived)= {cav_M_dm_fixed:.3e} M☉")
	print(f"    M_DM / M_bary = {cav_M_dm_fixed / model_cwdm_fixed.total_baryonic_mass():.2f}")

	# ─── Four-way comparison (standard models) ──────────────────────────
	print("\n" + "="*70)
	print("  FOUR-WAY COMPARISON: NFW vs BURKERT vs CAVITY vs CAVITY+WDM")
	print("="*70)

	n_data = len(data['r'])
	R_sun = 8.122
	n_dm = {'nfw': 2, 'burkert': 2, 'cavity': 4, 'cavity_wdm': 6}
	n_bary = len(BARYON_PARAM_NAMES)
	models = [
		('NFW', model_nfw, result_nfw, 'nfw'),
		('Burkert', model_burkert, result_burkert, 'burkert'),
		('Cavity', model_cavity, result_cavity, 'cavity'),
		('Cavity+WDM', model_cwdm, result_cwdm, 'cavity_wdm'),
	]

	print(f"\n  {'Property':30s}  {'NFW':>15s}  {'Burkert':>15s}  {'Cavity':>15s}  {'Cavity+WDM':>15s}")
	print(f"  {'-'*30}  {'-'*15}  {'-'*15}  {'-'*15}  {'-'*15}")
	for prop, fn in [
		('χ²', lambda m, r, k: f"{r.fun:15.1f}"),
		('χ²_red', lambda m, r, k: f"{r.fun / max(n_data - n_bary - n_dm[k], 1):15.2f}"),
		('N_free (DM)', lambda m, r, k: f"{n_dm[k]:15d}"),
		('Total baryonic [M☉]', lambda m, r, k: f"{m.total_baryonic_mass():15.3e}"),
		('DM mass (200 kpc) [M☉]', lambda m, r, k: f"{m.dm_virial_mass():15.3e}"),
	]:
		row = f"  {prop:30s}"
		for label, mdl, res, key in models:
			row += fn(mdl, res, key)
		print(row)

	row = f"  {'Baryon fraction':30s}"
	for label, mdl, res, key in models:
		f_b = mdl.total_baryonic_mass() / mdl.dm_virial_mass()
		row += f"{f_b:15.3f}"
	print(row)

	msun_kpc3_to_gev_cm3 = 2.63e-8
	for label, mdl, res, key in models:
		rho_local = mdl.dm_density(np.array([R_sun]))[0]
		print(f"  {'ρ_DM(R☉) ' + label:30s}  {rho_local * msun_kpc3_to_gev_cm3:.3f} GeV/cm³")
	print(f"  {'Eilers+ 2019':30s}  0.30 ± 0.03 GeV/cm³")

	# ─── Cavity ρ_bg comparison ──────────────────────────────────────────
	print(f"\n{'='*70}")
	print(f"  CAVITY ρ_bg PARAMETRIZATION COMPARISON")
	print(f"{'='*70}")
	print(f"\n  {'':30s}  {'M_DM free':>15s}  {'ρ_bg free':>15s}  {'ρ_bg fixed':>15s}")
	print(f"  {'-'*30}  {'-'*15}  {'-'*15}  {'-'*15}")
	print(f"  {'χ²':30s}  {result_cwdm.fun:15.1f}  {result_cwdm_rho.fun:15.1f}  {result_cwdm_fixed.fun:15.1f}")
	n_free_mdm = n_bary + 6
	n_free_rho = n_bary + 6
	n_free_fix = n_bary + 5  # cav_rho_bg excluded
	print(f"  {'χ²_red':30s}  {result_cwdm.fun / max(n_data - n_free_mdm, 1):15.2f}"
		  f"  {result_cwdm_rho.fun / max(n_data - n_free_rho, 1):15.2f}"
		  f"  {result_cwdm_fixed.fun / max(n_data - n_free_fix, 1):15.2f}")
	print(f"  {'ρ_bg [M☉/kpc³]':30s}  {rho_bg_cwdm:15.0f}  {p_rho_free['cav_rho_bg']:15.0f}  {p_rho_fix['cav_rho_bg']:15.0f}")
	print(f"  {'ρ_bg / ρ_cosmo':30s}  {rho_bg_cwdm / RHO_BG:15.0f}×  {p_rho_free['cav_rho_bg'] / RHO_BG:15.0f}×  {p_rho_fix['cav_rho_bg'] / RHO_BG:15.0f}×")
	print(f"  {'Cavity M_DM [M☉]':30s}  {p_cwdm['cav_M_DM']:15.3e}  {cav_M_dm_free:15.3e}  {cav_M_dm_fixed:15.3e}")
	print(f"  {'R_halo [kpc]':30s}  {p_cwdm['cav_R_halo']:15.0f}  {p_rho_free['cav_R_halo']:15.0f}  {p_rho_fix['cav_R_halo']:15.0f}")
	print(f"  {'α':30s}  {p_cwdm['cav_alpha']:15.3f}  {p_rho_free['cav_alpha']:15.3f}  {p_rho_fix['cav_alpha']:15.3f}")
	print(f"  {'R_s':30s}  {p_cwdm['cav_R_s']:15.4f}  {p_rho_free['cav_R_s']:15.4f}  {p_rho_fix['cav_R_s']:15.4f}")
	print(f"  {'WDM ρ₀ [M☉/kpc³]':30s}  {p_cwdm['wdm_rho_0']:15.3e}  {p_rho_free['wdm_rho_0']:15.3e}  {p_rho_fix['wdm_rho_0']:15.3e}")
	print(f"  {'WDM r₀ [kpc]':30s}  {p_cwdm['wdm_r_0']:15.3f}  {p_rho_free['wdm_r_0']:15.3f}  {p_rho_fix['wdm_r_0']:15.3f}")
	print(f"  {'Total baryonic [M☉]':30s}  {model_cwdm.total_baryonic_mass():15.3e}  {model_cwdm_rho.total_baryonic_mass():15.3e}  {model_cwdm_fixed.total_baryonic_mass():15.3e}")
	for lbl, mdl in [('M_DM free', model_cwdm), ('ρ_bg free', model_cwdm_rho), ('ρ_bg fixed', model_cwdm_fixed)]:
		rho_local = mdl.dm_density(np.array([R_sun]))[0]
		print(f"  {'ρ_DM(R☉) ' + lbl:30s}  {rho_local * msun_kpc3_to_gev_cm3:.3f} GeV/cm³")

	# ─── Plot with Burkert as primary (decomposition), NFW as alt ───────
	#plot_decomposition(data, model_burkert, dm_profile_burkert,
	#				   model_alt=model_nfw, dm_profile_alt=dm_profile_nfw)
	
	# ─── Plot with Cavity as primary (decomposition), Burkert as alt ───────
	plot_decomposition(data, model_cavity, dm_profile_cavity,
					   model_alt=model_burkert, dm_profile_alt=dm_profile_burkert)

	# ─── Additional: Cavity+WDM decomposition plot ──────────────────────
	plot_cavity_wdm(data, model_cwdm, dm_profile_cwdm,
					model_nfw=model_nfw, model_burkert=model_burkert)
	#plot_cavity_wdm(data, model_cwdm_rho, dm_profile_cwdm_rho,
	#				model_nfw=model_nfw, model_burkert=model_burkert)
