"""
Physical unit system for AntiEngine galactic simulations.

We use the following base units:
  Length : 1 kpc  = 3.0857e19 m
  Mass   : 1 M☉   = 1.9885e30 kg
  Time   : 1 Gyr  = 3.1558e16 s

In these units, Newton's G works out to:
  G = 6.674e-11 N m² kg⁻² * (1 M☉ / 1.9885e30 kg) / (1 kpc / 3.0857e19 m)³ * (1 Gyr / 3.1558e16 s)²
  G ≈ 4.4985e-6  kpc³ M☉⁻¹ Gyr⁻²

This is the value to use in all galactic simulations. Velocities in these units are
  1 kpc/Gyr ≈ 0.9778 km/s  →  1 km/s ≈ 1.0227 kpc/Gyr

Constants are computed at import time and should be treated as read-only.
"""

# --------------------------------------------------------------------------- #
# SI base values
# --------------------------------------------------------------------------- #
_M_sun_kg   = 1.98848e30     # kg  — IAU 2015 nominal solar mass
_kpc_m      = 3.08568e19     # m   — parsec = 3.08568e16 m  → 1 kpc = 1e3 pc
_Gyr_s      = 3.15576e16     # s   — Julian Gyr = 365.25 * 24 * 3600 * 1e9 s
_G_SI       = 6.67430e-11    # m³ kg⁻¹ s⁻²

# --------------------------------------------------------------------------- #
# G in simulation units  [ kpc³  M☉⁻¹  Gyr⁻² ]
# --------------------------------------------------------------------------- #
G_kpc_Msun_Gyr = _G_SI * _M_sun_kg / _kpc_m**3 * _Gyr_s**2

# --------------------------------------------------------------------------- #
# Convenient conversion factors
# --------------------------------------------------------------------------- #
# 1 km/s in kpc/Gyr
kms_to_kpc_per_Gyr  = _Gyr_s / (_kpc_m * 1e-3)

# 1 kpc/Gyr in km/s
kpc_per_Gyr_to_kms  = 1.0 / kms_to_kpc_per_Gyr

# 1 ly in kpc  (1 ly = 0.30660 pc)
ly_to_kpc  = 0.30660e-3

# --------------------------------------------------------------------------- #
# Reference galaxy parameters (Milky Way / Andromeda scale)
# --------------------------------------------------------------------------- #
# Stellar mass range (Milky Way: 46–64 Gyr M☉; Andromeda: 100–150 billion M☉)
MW_STELLAR_MASS_MSUN   = 55e9     # M☉  — midpoint of Milky Way estimate
AND_STELLAR_MASS_MSUN  = 125e9    # M☉  — midpoint of Andromeda estimate

# Andromeda component fractions
AND_BULGE_FRACTION = 0.30
AND_DISK_FRACTION  = 0.56
AND_HALO_FRACTION  = 0.14

# Gas mass ≈ 10–15% of stellar mass
GAS_TO_STELLAR_FRACTION = 0.125

# Andromeda disk diameter ≈ 46.56 kpc → disk scale radius r_d ≈ diameter / (4π * e^-1) ≈ 3-4 kpc
# Use exponential disk: Σ(r) ∝ exp(-r/r_d), half-mass radius ≈ 1.7 r_d
AND_DISK_RADIUS_KPC        = 23.28    # kpc  — disk half-diameter
AND_DISK_SCALE_RADIUS_KPC  = 4.0      # kpc  — exponential scale radius r_d
AND_BULGE_SCALE_RADIUS_KPC = 1.0      # kpc  — Hernquist scale radius
AND_STELLAR_HALO_RADIUS_KPC = 33.7    # kpc  — stellar halo half-diameter (67.45 kpc diameter)
AND_GAS_HALO_RADIUS_KPC    = 100.0    # kpc  — gas halo (hundreds of kly ≈ ~100 kpc)


def print_units():
	"""Print the unit system for verification."""
	print("=== AntiEngine unit system ===")
	print(f"  Base units:  kpc, M\u2609, Gyr")
	print(f"  G = {G_kpc_Msun_Gyr:.6e}  kpc\u00b3 M\u2609\u207b\u00b9 Gyr\u207b\u00b2")
	print(f"  1 km/s  = {kms_to_kpc_per_Gyr:.4f} kpc/Gyr")
	print(f"  1 kpc/Gyr = {kpc_per_Gyr_to_kms:.4f} km/s")
	print(f"  Circular velocity check: v_c(r=8.5 kpc, M_enc=6e10 M\u2609)")
	M_enc = 6e10
	r     = 8.5
	v_c   = (G_kpc_Msun_Gyr * M_enc / r) ** 0.5  # kpc/Gyr
	print(f"    = {v_c:.4f} kpc/Gyr  = {v_c * kpc_per_Gyr_to_kms:.1f} km/s  (expect ~220 km/s)")
