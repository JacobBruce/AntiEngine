#!/usr/bin/env python3
"""
Phase 4: Gravitational lensing test — cavity model vs standard DM profiles.

Computes and compares the excess surface density (ESD) ΔΣ(R) and rotation
curves for a complete galaxy model (SMBH + bulge + disk + stellar halo +
gas halo) combined with:
  1. NFW dark matter halo (standard ΛCDM)
  2. Cavity model (anti-universe neg-mass deficit, uniform α=0)
  3. Burkert profile (cored alternative)

The key insight: a uniform cavity (α=0) naturally solves the cuspy halo
problem, while the complete baryonic model provides the inner uplift needed
for realistic rotation curves. No cuspy dark matter profile is required.

Physical units: kpc, M☉ throughout.
"""

import numpy as np
import matplotlib.pyplot as plt

from simulations.lensing.profiles import (
	NFWProfile, BurkertProfile, PseudoIsothermalProfile,
	ExponentialDiskProfile, HernquistProfile, CavityProfile,
	PointMassProfile, CompositeProfile,
)
from simulations.lensing.lensing import sigma_crit_from_redshifts
from antiengine.units import (
	G_kpc_Msun_Gyr, kpc_per_Gyr_to_kms,
	AND_STELLAR_MASS_MSUN, AND_DISK_FRACTION, AND_BULGE_FRACTION, AND_HALO_FRACTION,
	AND_DISK_SCALE_RADIUS_KPC, AND_BULGE_SCALE_RADIUS_KPC,
	AND_STELLAR_HALO_RADIUS_KPC, AND_GAS_HALO_RADIUS_KPC,
	GAS_TO_STELLAR_FRACTION,
)


# ─────────────────────────────────────────────────────────────────────────────
# Galaxy parameters (Andromeda-like, consistent with Phase 2A)
# ─────────────────────────────────────────────────────────────────────────────

M_stellar = AND_STELLAR_MASS_MSUN   # 125e9 M☉
M_200     = 1.3e12                  # M☉ — total virial mass (NFW reference)
c_200     = 10.0                    # NFW concentration parameter
bh_mass   = 140e6                   # M☉ — SMBH (M31*)

# Component fractions (of stellar mass)
disk_frac  = AND_DISK_FRACTION        # 0.56
bulge_frac = AND_BULGE_FRACTION       # 0.30
halo_frac  = AND_HALO_FRACTION        # 0.14
gas_ratio  = GAS_TO_STELLAR_FRACTION  # 0.125

# Component masses
M_disk  = M_stellar * disk_frac      # 70.0e9 M☉
M_bulge = M_stellar * bulge_frac     # 37.5e9 M☉
M_shalo = M_stellar * halo_frac      # 17.5e9 M☉
M_gas   = M_stellar * gas_ratio      # 15.625e9 M☉
M_bary  = M_disk + M_bulge + M_shalo + M_gas + bh_mass  # ~140.8e9 M☉

# Scale parameters (kpc)
r_d        = AND_DISK_SCALE_RADIUS_KPC       # 4.0 kpc — disk scale radius
bulge_a    = AND_BULGE_SCALE_RADIUS_KPC      # 1.0 kpc — Hernquist bulge scale
shalo_rc   = 10.0                            # kpc — stellar halo core radius
shalo_rmax = AND_STELLAR_HALO_RADIUS_KPC     # 33.7 kpc — stellar halo truncation
gas_rc     = 5.0                             # kpc — gas halo core radius (β-model)
gas_rmax   = AND_GAS_HALO_RADIUS_KPC         # 100.0 kpc — gas halo truncation

# Cavity model parameters
# M_DM_cav: total effective DM mass in the cavity (= M_200 - M_bary)
# R_halo: transition radius where neg-mass density recovers to background
M_DM_cav  = M_200 - M_bary    # ≈ 1.16e12 M☉
R_halo    = 200.0             # kpc — comparable to r_200
Rs_cav    = 0.05
alpha_cav = 2.5

cav_label = f"Cavity (α={alpha_cav}, Rs={Rs_cav})"

print("=== Galaxy Model (Andromeda-like) ===")
print(f"  SMBH:           {bh_mass:.2e} M☉")
print(f"  Bulge:          {M_bulge:.2e} M☉  (a={bulge_a} kpc)")
print(f"  Disk:           {M_disk:.2e} M☉  (r_d={r_d} kpc)")
print(f"  Stellar halo:   {M_shalo:.2e} M☉  (r_c={shalo_rc}, r_max={shalo_rmax} kpc)")
print(f"  Gas halo:       {M_gas:.2e} M☉  (r_c={gas_rc}, r_max={gas_rmax} kpc)")
print(f"  Total baryonic: {M_bary:.2e} M☉")
print(f"  Cavity DM:      {M_DM_cav:.2e} M☉  (R_halo={R_halo} kpc)")
print(f"  M_200 (NFW):    {M_200:.2e} M☉")


# ─────────────────────────────────────────────────────────────────────────────
# Build profiles
# ─────────────────────────────────────────────────────────────────────────────

# --- Baryonic components ---
smbh  = PointMassProfile(M=bh_mass, label="SMBH")
bulge = HernquistProfile(M=M_bulge, a=bulge_a, label="Bulge")
disk  = ExponentialDiskProfile(M_d=M_disk, r_d=r_d, label="Disk")

# Stellar halo — pseudo-isothermal ρ(r) = ρ₀ / (1 + (r/r_c)²)
# Normalise ρ₀ so total mass within r_max equals M_shalo:
#   M = 4πρ₀r_c³(x - arctan(x)), x = r_max/r_c
x_sh = shalo_rmax / shalo_rc
shalo_rho0 = M_shalo / (4 * np.pi * shalo_rc**3 * (x_sh - np.arctan(x_sh)))
stellar_halo = PseudoIsothermalProfile(
	rho_0=shalo_rho0, r_c=shalo_rc, r_max=shalo_rmax, label="Stellar halo",
)

# Gas halo — pseudo-isothermal (β-model with β=2/3 → same functional form)
x_g = gas_rmax / gas_rc
gas_rho0 = M_gas / (4 * np.pi * gas_rc**3 * (x_g - np.arctan(x_g)))
gas_halo = PseudoIsothermalProfile(
	rho_0=gas_rho0, r_c=gas_rc, r_max=gas_rmax, label="Gas halo",
)

print(f"\n  Stellar halo ρ₀ = {shalo_rho0:.2e} M☉/kpc³")
print(f"  Gas halo ρ₀     = {gas_rho0:.2e} M☉/kpc³")

# Baryonic composites
baryons_all = CompositeProfile([smbh, bulge, disk, stellar_halo, gas_halo], label="All baryons")

# --- Dark matter profiles ---
nfw = NFWProfile(M_200=M_200, c=c_200)
print(f"\n  NFW: r_200={nfw.r_200:.1f} kpc, r_s={nfw.r_s:.1f} kpc, M(r_200)={nfw.M_enc(nfw.r_200):.2e} M☉")

cav = CavityProfile(M_DM=M_DM_cav, R_halo=R_halo, profile_type='logistic',
					 alpha=alpha_cav, R_s=Rs_cav, label="Cavity (logistic)")
print(f"\n  Cavity: R_halo={R_halo:.0f} kpc, ρ_bg={cav.rho_bg:.0f} M☉/kpc³, "
	  f"profile={cav.profile_type}, a={cav.alpha}, s={cav.R_s}, "
	  f"M_enc(R_halo)={cav.M_enc(R_halo):.2e} M☉")

# Burkert — calibrate ρ₀ to match NFW M_enc at 100 kpc
from scipy.optimize import brentq

target_M100 = float(nfw.M_enc(100.0))
r_0_burk = 8.0

def burk_M100_residual(log_rho0):
	b = BurkertProfile(rho_0=10**log_rho0, r_0=r_0_burk, r_max=1000.0)
	return float(b.M_enc(100.0)) - target_M100

rho0_burk = 10 ** brentq(burk_M100_residual, 4, 10)
burk = BurkertProfile(rho_0=rho0_burk, r_0=r_0_burk, r_max=1000.0)
print(f"  Burkert: ρ₀={rho0_burk:.2e} M☉/kpc³, r_0={r_0_burk} kpc")

# Pseudo-isothermal — fit ρ_0 to match NFW M_enc at 100 kpc
r_c_piso = 5.0  # kpc

def piso_M100_residual(log_rho0):
	p = PseudoIsothermalProfile(rho_0=10**log_rho0, r_c=r_c_piso, r_max=1000.0)
	return float(p.M_enc(100.0)) - target_M100

rho0_piso = 10 ** brentq(piso_M100_residual, 4, 10)
piso = PseudoIsothermalProfile(rho_0=rho0_piso, r_c=r_c_piso, r_max=1000.0)
print(f"Pseudo-iso: ρ_0={rho0_piso:.2e} M☉/kpc³, r_c={r_c_piso} kpc, M(100)={piso.M_enc(100.0):.2e} M☉")

# ─────────────────────────────────────────────────────────────────────────────
# Rotation curves (the key diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

r_rc = np.linspace(0.5, 200, 400)

# Individual baryonic components
M_smbh_enc  = smbh.M_enc(r_rc)
M_bulge_enc = bulge.M_enc(r_rc)
M_disk_enc  = disk.M_enc(r_rc)
M_shalo_enc = stellar_halo.M_enc(r_rc)
M_gas_enc   = gas_halo.M_enc(r_rc)
M_bary_all  = M_smbh_enc + M_bulge_enc + M_disk_enc + M_shalo_enc + M_gas_enc

def v_circ(M_enc, r):
	return np.sqrt(G_kpc_Msun_Gyr * M_enc / r) * kpc_per_Gyr_to_kms

v_smbh     = v_circ(M_smbh_enc, r_rc)
v_bulge    = v_circ(M_bulge_enc, r_rc)
v_disk     = v_circ(M_disk_enc, r_rc)
v_shalo    = v_circ(M_shalo_enc, r_rc)
v_gas      = v_circ(M_gas_enc, r_rc)
v_bary_all = v_circ(M_bary_all, r_rc)

# DM profiles (all with full baryons)
v_nfw      = v_circ(M_bary_all + nfw.M_enc(r_rc), r_rc)
v_burk     = v_circ(M_bary_all + burk.M_enc(r_rc), r_rc)
v_cav      = v_circ(M_bary_all + cav.M_enc(r_rc), r_rc)
v_piso     = v_circ(M_bary_all + piso.M_enc(r_rc), r_rc)

# cavity profile without baryons
v_cav_only = v_circ(cav.M_enc(r_rc), r_rc)


# ─────────────────────────────────────────────────────────────────────────────
# Compute ESD for composite models (baryons + DM)
# ─────────────────────────────────────────────────────────────────────────────

R = np.logspace(np.log10(5), np.log10(600), 30)

print("\nComputing ESD profiles (this may take a minute)...")

# DM-only ESD (reveals true model differences, unmasked by baryons)
esd_nfw_only   = nfw.delta_sigma(R)
print("  NFW (DM only) done")
esd_cav_only = cav.delta_sigma(R)
print("  Cavity (DM only) done")
esd_burk_only  = burk.delta_sigma(R)
print("  Burkert (DM only) done")
esd_piso_only = piso.delta_sigma(R)
print("  Pseudo-isothermal done")

# Full composites for ESD comparison (what observers actually measure)
comp_nfw   = CompositeProfile([baryons_all, nfw], label="Baryons + NFW")
comp_cav   = CompositeProfile([baryons_all, cav], label="Baryons + Cavity")
comp_piso  = CompositeProfile([baryons_all, piso], label="Baryons + Pseudo-isothermal")
comp_burk  = CompositeProfile([baryons_all, burk], label="Baryons + Burkert")

esd_comp_nfw   = comp_nfw.delta_sigma(R)
print("  NFW + baryons done")
esd_comp_cav = comp_cav.delta_sigma(R)
print("  Cavity + baryons done")
esd_comp_burk  = comp_burk.delta_sigma(R)
print("  Burkert + baryons done")
esd_comp_piso = comp_piso.delta_sigma(R)
print("  Pseudo-isothermal + baryons done")
esd_baryons = baryons_all.delta_sigma(R)
print("  Baryons-only done")


# ─────────────────────────────────────────────────────────────────────────────
# Convergence and shear
# ─────────────────────────────────────────────────────────────────────────────

z_l, z_s = 0.2, 0.6
Sigma_cr = sigma_crit_from_redshifts(z_l, z_s)
print(f"\nΣ_crit(z_l={z_l}, z_s={z_s}) = {Sigma_cr:.3e} M☉/kpc²")

# Composite (baryons + DM) — what observers measure
kappa_comp_nfw   = comp_nfw.sigma(R) / Sigma_cr
gamma_comp_nfw   = esd_comp_nfw / Sigma_cr
kappa_comp_cav   = comp_cav.sigma(R) / Sigma_cr
gamma_comp_cav   = esd_comp_cav / Sigma_cr

# DM-only — reveals the actual model difference
kappa_nfw_only   = nfw.sigma(R) / Sigma_cr
gamma_nfw_only   = esd_nfw_only / Sigma_cr
kappa_cav_only   = cav.sigma(R) / Sigma_cr
gamma_cav_only   = esd_cav_only / Sigma_cr


# ─────────────────────────────────────────────────────────────────────────────
# Inner density profiles (cuspy halo diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

r_inner = np.logspace(np.log10(0.5), np.log10(20), 100)
r_full  = np.logspace(np.log10(0.5), np.log10(500), 200)
rho_nfw_inner   = nfw.rho(r_inner)
rho_cav_inner   = cav.rho(r_inner)
rho_burk_inner  = burk.rho(r_inner)
rho_piso_inner  = piso.rho(r_inner)
rho_nfw_full    = nfw.rho(r_full)
rho_cav_full    = cav.rho(r_full)
rho_burk_full   = burk.rho(r_full)
rho_piso_full   = piso.rho(r_full)

def log_slope(r_arr, rho_arr, r_eval=1.0):
	"""d(log ρ)/d(log r) at r_eval via finite difference."""
	lr = np.log10(r_arr)
	lrho = np.log10(np.maximum(rho_arr, 1e-30))
	idx = np.argmin(np.abs(r_arr - r_eval))
	if idx == 0:
		idx = 1
	return (lrho[idx] - lrho[idx-1]) / (lr[idx] - lr[idx-1])

print("\n=== Inner density slope at r=1 kpc ===")
print(f"  NFW:               {log_slope(r_inner, rho_nfw_inner):.2f}  (expect -1)")
print(f"  Cavity ({cav.profile_type}): {log_slope(r_inner, rho_cav_inner):.2f}")
print(f"  Burkert:           {log_slope(r_inner, rho_burk_inner):.2f}  (expect 0)")
print(f"  Pseudo-isothermal: {log_slope(r_inner, rho_piso_inner):.2f}  (expect 0)")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor('#111111')
fig.suptitle(
	"Phase 4: Gravitational Lensing — Cavity Model vs Standard DM Profiles",
	fontsize=13, fontweight='bold', color='white',
)
for ax in axes.ravel():
	ax.set_facecolor('#0d0d0d')
	ax.tick_params(colors='#aaaaaa')
	ax.xaxis.label.set_color('#cccccc')
	ax.yaxis.label.set_color('#cccccc')
	ax.title.set_color('#ffffff')
	for spine in ax.spines.values():
		spine.set_edgecolor('#333333')

# --- Panel 1: Rotation curve decomposition (the key result) ---
ax = axes[0, 0]
ax.plot(r_rc, v_smbh, ':', lw=1, color='#666666', label='SMBH')
ax.plot(r_rc, v_bulge, '--', lw=1.2, color='#CC9900', label='Bulge')
ax.plot(r_rc, v_disk, '--', lw=1.2, color='#00CCCC', label='Disk')
ax.plot(r_rc, v_shalo, '--', lw=1.2, color='#33CCFF', label='Stellar halo')
ax.plot(r_rc, v_gas, '--', lw=1.2, color='#66FF99', label='Gas halo')
ax.plot(r_rc, v_bary_all, '-', lw=1.5, color='#dddddd', label='All baryons')
ax.plot(r_rc, v_cav_only, 'r--', lw=1.5, label=cav_label)
ax.plot(r_rc, v_cav, '-', lw=2.5, color='#4488ff', label='Baryons + Cavity')
ax.set_xlabel('Radius r [kpc]')
ax.set_ylabel('v_c [km/s]')
ax.set_title('Rotation Curve Decomposition')
ax.legend(fontsize=7, loc='upper right', facecolor='#222222', labelcolor='white', edgecolor='#555555')
ax.set_xlim(0, 200)
ax.set_ylim(0, 300)
ax.grid(True, alpha=0.25, color='#555555')

# --- Panel 2: Rotation curve comparison (DM models) ---
ax = axes[0, 1]
ax.plot(r_rc, v_cav, '-', lw=2, color='red', label=cav_label)
ax.plot(r_rc, v_nfw, 'b-', lw=2, label='NFW')
ax.plot(r_rc, v_burk, 'g--', lw=1.5, label='Burkert')
ax.plot(r_rc, v_piso, 'm:', lw=1.5, label='Pseudo-isothermal')
ax.plot(r_rc, v_bary_all, ':', lw=1, color='#888888', alpha=0.7, label='Baryons only')
ax.set_xlabel('Radius r [kpc]')
ax.set_ylabel('v_c [km/s]')
ax.set_title('Rotation Curves — DM Model Comparison')
ax.legend(fontsize=7, loc='upper right', facecolor='#222222', labelcolor='white', edgecolor='#555555')
ax.set_xlim(0, 200)
ax.set_ylim(0, 300)
ax.grid(True, alpha=0.25, color='#555555')

# --- Panel 3: Excess Surface Density ---
ax = axes[1, 0]
ax.loglog(R, np.maximum(esd_cav_only, 1), 'r-', lw=2, label=cav_label)
ax.loglog(R, esd_nfw_only, 'b-', lw=2, label='NFW')
ax.loglog(R, esd_burk_only, 'g--', lw=1.5, label='Burkert')
ax.loglog(R, esd_piso_only, 'm:', lw=1.5, label='Pseudo-isothermal')
ax.loglog(R, np.maximum(esd_baryons, 1), ':', lw=1, color='#888888', alpha=0.7, label='Baryons only')
ax.set_xlabel('Projected radius R [kpc]')
ax.set_ylabel('ΔΣ(R) [M☉/kpc²]')
ax.set_title('Excess Surface Density (lensing observable)')
ax.legend(fontsize=7, loc='lower left', facecolor='#222222', labelcolor='white', edgecolor='#555555')
ax.set_xlim(5, 600)
ax.set_ylim(1e3, 3e8)
ax.grid(True, alpha=0.25, color='#555555')

# --- Panel 4: Convergence & shear ---
ax = axes[1, 1]
ax.loglog(R, kappa_nfw_only, 'b-', lw=2, label='κ NFW')
ax.loglog(R, np.maximum(kappa_cav_only, 1e-10), 'r-', lw=2, label='κ Cavity')
ax.loglog(R, gamma_nfw_only, 'b--', lw=1.5, label='γ_t NFW')
ax.loglog(R, np.maximum(gamma_cav_only, 1e-10), 'r--', lw=1.5, label='γ_t Cavity')
ax.set_xlabel('Projected radius R [kpc]')
ax.set_ylabel('κ, γ_t (dimensionless)')
ax.set_title(f'Convergence & Shear (z_l={z_l}, z_s={z_s})')
ax.legend(fontsize=8, loc='lower left', facecolor='#222222', labelcolor='white', edgecolor='#555555')
ax.set_xlim(5, 600)
ax.grid(True, alpha=0.25, color='#555555')

# --- Panel 5: ESD ratio to NFW (true model difference) ---
ax = axes[0, 2]
ratio_dm_cav  = np.where(esd_nfw_only > 0, esd_cav_only / esd_nfw_only, np.nan)
ratio_dm_burk = np.where(esd_nfw_only > 0, esd_burk_only / esd_nfw_only, np.nan)
ratio_dm_piso = np.where(esd_nfw_only > 0, esd_piso_only / esd_nfw_only, np.nan)

ax.semilogx(R, ratio_dm_cav, 'r-', lw=2, label=cav_label)
ax.semilogx(R, ratio_dm_burk, 'g--', lw=1.5, label='Burkert')
ax.semilogx(R, ratio_dm_piso, 'm:', lw=1.5, label='Pseudo-isothermal')
ax.axhline(1.0, color='#4488ff', ls='-', lw=1, alpha=0.5)
ax.fill_between(R, 0.7, 1.3, color='#4488ff', alpha=0.08, label='±30% obs. uncertainty')
ax.fill_between(R, 0.8, 1.2, color='#4488ff', alpha=0.08, label='±20% obs. uncertainty')
ax.set_xlabel('Projected radius R [kpc]')
ax.set_ylabel('ΔΣ / ΔΣ_NFW')
ax.set_title('ESD Ratio to NFW (detectability)')
ax.legend(fontsize=7, loc='upper right', facecolor='#222222', labelcolor='white', edgecolor='#555555')
ax.set_xlim(5, 600)
ax.set_ylim(0, 2.0)
ax.grid(True, alpha=0.25, color='#555555')

# --- Panel 6: 3D DM density profiles (full range showing cavity transition) ---
ax = axes[1, 2]
ax.loglog(r_full, np.maximum(rho_cav_full, 1e-30), 'r-', lw=2, label=cav_label)
ax.loglog(r_full, rho_nfw_full, 'b-', lw=2, label='NFW')
ax.loglog(r_full, rho_burk_full, 'g--', lw=1.5, label='Burkert')
ax.loglog(r_full, rho_piso_full, 'm:', lw=1.5, label='Pseudo-isothermal')
ax.axvline(R_halo, color='red', ls=':', alpha=0.4, lw=0.8)
#ax.text(R_halo * 0.7, 1e2, f'R_halo={R_halo:.0f}', fontsize=6, color='#ff6666', rotation=90)
ax.set_xlabel('3D radius r [kpc]')
ax.set_ylabel('ρ(r) [M☉/kpc³]')
ax.set_title('DM Density Profile (full range)')
ax.legend(fontsize=7, loc='lower left', facecolor='#222222', labelcolor='white', edgecolor='#555555')
ax.grid(True, alpha=0.25, color='#555555')

plt.tight_layout()
plt.savefig('lensing_comparison.png', dpi=150, bbox_inches='tight', facecolor='#111111')
print("\nSaved: lensing_comparison.png")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("SUMMARY: Complete Galaxy + Cavity Model")
print("=" * 80)

print("\n--- Galaxy Mass Budget ---")
print(f"  SMBH:           {bh_mass:.2e} M☉  ({bh_mass/M_bary*100:.2f}%)")
print(f"  Bulge:          {M_bulge:.2e} M☉  ({M_bulge/M_bary*100:.1f}%)")
print(f"  Disk:           {M_disk:.2e} M☉  ({M_disk/M_bary*100:.1f}%)")
print(f"  Stellar halo:   {M_shalo:.2e} M☉  ({M_shalo/M_bary*100:.1f}%)")
print(f"  Gas halo:       {M_gas:.2e} M☉  ({M_gas/M_bary*100:.1f}%)")
print(f"  Total baryonic: {M_bary:.2e} M☉")
print(f"  Cavity DM:      {M_DM_cav:.2e} M☉  (R_halo={R_halo} kpc)")
print(f"  Cavity ρ_bg:    {cav.rho_bg:.0f} M☉/kpc³")
print(f"  DM/bary ratio:  {M_DM_cav/M_bary:.1f}")

print("\n--- Rotation Curve at Key Radii ---")
for r_eval in [5, 10, 20, 50, 100]:
	idx = np.argmin(np.abs(r_rc - r_eval))
	print(f"  r={r_eval:3d} kpc: "
		  f"baryons={v_bary_all[idx]:.0f}, "
		  f"cavity={v_cav[idx]:.0f}, "
		  f"NFW={v_nfw[idx]:.0f}, "
		  f"Burkert={v_burk[idx]:.0f} km/s")

print("\n--- Cavity Model Summary ---")
print(f"R_halo = {R_halo:.0f} kpc, ρ_bg = {cav.rho_bg:.0f} M☉/kpc³")
print(f"Profile: {cav.profile_type}, a={cav.alpha}, b={cav.R_s}")
print(f"Deficit goes from ρ_bg at center to 0 at R_halo={R_halo:.0f} kpc.")

print("\n--- Lensing Detectability (DM-only ESD) ---")
R_eval = np.array([50, 100, 200])
for r in R_eval:
	idx = np.argmin(np.abs(R - r))
	if esd_nfw_only[idx] > 0:
		dev = abs(1 - esd_cav_only[idx] / esd_nfw_only[idx]) * 100
		det = "YES" if dev > 30 else ("MARGINAL" if dev > 20 else "NO")
		print(f"  R={r:3d} kpc: Cavity deviates {dev:.0f}% from NFW ({det})")
	else:
		print(f"  R={r:3d} kpc: NFW ESD ≈ 0, cannot compare")

print("\n--- Lensing Detectability (composite: baryons + DM) ---")
print("Note: total signal incl. baryons — what observers actually measure.")
print("Hernquist bulge has 1/R projected tail dominating ESD at all R.")
for r in R_eval:
	idx = np.argmin(np.abs(R - r))
	dev = abs(1 - esd_comp_cav[idx] / esd_comp_nfw[idx]) * 100
	det = "YES" if dev > 30 else ("MARGINAL" if dev > 20 else "NO")
	print(f"  R={r:3d} kpc: Cavity deviates {dev:.0f}% from NFW ({det})")
