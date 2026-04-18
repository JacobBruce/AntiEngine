"""
Gravitational lensing computations.

Computes convergence κ, tangential shear γ_t, and excess surface density ΔΣ
from mass profiles. These are the key observables in galaxy-galaxy lensing.

Key relations:
	κ(R)  = Σ(R) / Σ_crit              — convergence
	γ_t(R) = ΔΣ(R) / Σ_crit            — tangential shear
	ΔΣ(R) = ⟨Σ⟩(<R) − Σ(R)            — excess surface density (ESD)

where Σ_crit = c² D_s / (4π G D_l D_ls) is the critical surface density.

Units: kpc, M☉, Gyr (consistent with AntiEngine)
"""

import numpy as np

from antiengine.units import G_kpc_Msun_Gyr


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants for lensing
# ─────────────────────────────────────────────────────────────────────────────

# Speed of light in simulation units: kpc / Gyr
_c_kms = 299792.458   # km/s
_kms_to_kpc_Gyr = 1.0227  # kpc/Gyr per km/s
C_KPC_GYR = _c_kms * _kms_to_kpc_Gyr   # ≈ 306601 kpc/Gyr


# ─────────────────────────────────────────────────────────────────────────────
# Critical surface density
# ─────────────────────────────────────────────────────────────────────────────

def sigma_crit(D_l, D_s, D_ls):
	"""
	Critical surface density for strong/weak lensing.

	Σ_crit = c² D_s / (4π G D_l D_ls)

	Parameters
	----------
	D_l  : angular diameter distance to lens [kpc]
	D_s  : angular diameter distance to source [kpc]
	D_ls : angular diameter distance from lens to source [kpc]

	Returns
	-------
	Σ_crit [M☉/kpc²]
	"""
	return C_KPC_GYR ** 2 * D_s / (4 * np.pi * G_kpc_Msun_Gyr * D_l * D_ls)


def sigma_crit_from_redshifts(z_l, z_s, H0=0.07159, Omega_m=0.3, Omega_DE=0.7):
	"""
	Σ_crit from lens and source redshifts, assuming flat ΛCDM for distances.

	Parameters
	----------
	z_l : lens redshift
	z_s : source redshift (must be > z_l)
	H0  : Hubble constant [Gyr⁻¹] (default: 70 km/s/Mpc)
	Omega_m  : matter density parameter
	Omega_DE : dark energy density parameter

	Returns
	-------
	Σ_crit [M☉/kpc²]
	"""
	from scipy import integrate as sp_int

	def E(z):
		return np.sqrt(Omega_m * (1 + z) ** 3 + Omega_DE)

	def comoving_dist(z1, z2):
		"""Comoving distance in kpc."""
		# d_C = (c/H0) ∫ dz/E(z)
		val, _ = sp_int.quad(lambda z: 1.0 / E(z), z1, z2)
		# Convert c/H0 from kpc/Gyr / Gyr⁻¹ = kpc... but need Mpc then to kpc
		d_H = C_KPC_GYR / H0  # Hubble distance [kpc]
		return d_H * val

	# Angular diameter distances (flat universe)
	d_C_l = comoving_dist(0, z_l)
	d_C_s = comoving_dist(0, z_s)
	d_C_ls = comoving_dist(z_l, z_s)

	D_l = d_C_l / (1 + z_l)
	D_s = d_C_s / (1 + z_s)
	D_ls = d_C_ls / (1 + z_s)

	return sigma_crit(D_l, D_s, D_ls)


# ─────────────────────────────────────────────────────────────────────────────
# Lensing observables from a mass profile
# ─────────────────────────────────────────────────────────────────────────────

def convergence(profile, R, Sigma_cr):
	"""
	Convergence κ(R) = Σ(R) / Σ_crit.

	Parameters
	----------
	profile  : MassProfile instance
	R        : projected radii [kpc] (scalar or array)
	Sigma_cr : critical surface density [M☉/kpc²]

	Returns
	-------
	κ(R) — dimensionless
	"""
	return profile.sigma(R) / Sigma_cr


def tangential_shear(profile, R, Sigma_cr):
	"""
	Tangential shear γ_t(R) = ΔΣ(R) / Σ_crit.

	This is the primary observable in galaxy-galaxy weak lensing.
	In the weak lensing regime (κ << 1), it equals the reduced shear g_t.

	Parameters
	----------
	profile  : MassProfile instance
	R        : projected radii [kpc]
	Sigma_cr : critical surface density [M☉/kpc²]

	Returns
	-------
	γ_t(R) — dimensionless
	"""
	return profile.delta_sigma(R) / Sigma_cr


def excess_surface_density(profile, R):
	"""
	Excess surface density ΔΣ(R) = ⟨Σ⟩(<R) − Σ(R) [M☉/kpc²].

	This is the quantity directly constrained by galaxy-galaxy lensing
	measurements, independent of source redshift distribution.

	Parameters
	----------
	profile : MassProfile instance
	R       : projected radii [kpc]

	Returns
	-------
	ΔΣ(R) [M☉/kpc²]
	"""
	return profile.delta_sigma(R)
