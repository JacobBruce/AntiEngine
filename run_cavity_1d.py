"""
Entry point for the 1D spherical cavity simulation.

Tests the cavity formation mechanism in spherical symmetry, tracking
Lagrangian neg-mass shells around a central galaxy.

Run from project root:
	python run_cavity_1d.py
"""

import numpy as np
import matplotlib.pyplot as plt
from simulations.cavity1d.sim import (
	Cavity1DConfig, run_cavity_1d,
	measure_density_profile, compute_rotation_curve_contribution,
)
from simulations.lensing.profiles import (
	NFWProfile, HernquistProfile, ExponentialDiskProfile,
	PseudoIsothermalProfile, PointMassProfile,
)
from antiengine.units import (
	G_kpc_Msun_Gyr, kpc_per_Gyr_to_kms,
	AND_STELLAR_MASS_MSUN, AND_DISK_FRACTION, AND_BULGE_FRACTION, AND_HALO_FRACTION,
	AND_DISK_SCALE_RADIUS_KPC, AND_BULGE_SCALE_RADIUS_KPC,
	AND_STELLAR_HALO_RADIUS_KPC, AND_GAS_HALO_RADIUS_KPC, GAS_TO_STELLAR_FRACTION,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
config = Cavity1DConfig(
	galaxy_mass     = 1.25e11,		# M☉ — Andromeda-scale
	rho_bg          = 3.9e4,		# M☉/kpc³
	r_domain        = 300.0,		# kpc
	n_shells        = 2000,
	dt              = 0.001,		# Gyr (1 Myr)
	n_steps         = 5000,			# 5 Gyr total
	record_every    = 50,
	reinject        = True,
	inflow_velocity = 0.0,			# stationary boundary
	velocity_dispersion = 0.0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Run both models: with and without Jeans swindle
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("MODEL 1: Jeans swindle (infinite medium, background-subtracted forces)")
print("=" * 70)
history_jeans = run_cavity_1d(config, use_jeans_swindle=True)

print("\n")
print("=" * 70)
print("MODEL 2: Raw forces (finite sphere, no background subtraction)")
print("=" * 70)
history_raw = run_cavity_1d(config, use_jeans_swindle=False)


# ─────────────────────────────────────────────────────────────────────────────
# Also run with thermal velocity dispersion
# ─────────────────────────────────────────────────────────────────────────────
config_warm = Cavity1DConfig(
	galaxy_mass     = 1.25e11,
	rho_bg          = 3.9e4,
	r_domain        = 300.0,
	n_shells        = 2000,
	dt              = 0.001,
	n_steps         = 5000,
	record_every    = 50,
	reinject        = True,
	inflow_velocity = 50.0 / kpc_per_Gyr_to_kms,		# 50 km/s inward
	velocity_dispersion = 100.0 / kpc_per_Gyr_to_kms,	# 100 km/s dispersion
)

print("\n")
print("=" * 70)
print("MODEL 3: Jeans swindle + warm neg mass (σ=100 km/s)")
print("=" * 70)
history_warm = run_cavity_1d(config_warm, use_jeans_swindle=True)


# ─────────────────────────────────────────────────────────────────────────────
# Build galaxy model for rotation curves
# ─────────────────────────────────────────────────────────────────────────────
M_stellar = AND_STELLAR_MASS_MSUN
M_disk    = M_stellar * AND_DISK_FRACTION
M_bulge   = M_stellar * AND_BULGE_FRACTION
M_shalo   = M_stellar * AND_HALO_FRACTION
M_gas     = M_stellar * GAS_TO_STELLAR_FRACTION
bh_mass   = 140e6

smbh  = PointMassProfile(M=bh_mass)
bulge = HernquistProfile(M=M_bulge, a=AND_BULGE_SCALE_RADIUS_KPC)
disk  = ExponentialDiskProfile(M_d=M_disk, r_d=AND_DISK_SCALE_RADIUS_KPC)
x_sh  = AND_STELLAR_HALO_RADIUS_KPC / 10.0
shalo_rho0 = M_shalo / (4 * np.pi * 10.0**3 * (x_sh - np.arctan(x_sh)))
stellar_halo = PseudoIsothermalProfile(rho_0=shalo_rho0, r_c=10.0, r_max=AND_STELLAR_HALO_RADIUS_KPC)
x_gas = AND_GAS_HALO_RADIUS_KPC / 5.0
gas_rho0 = M_gas / (4 * np.pi * 5.0**3 * (x_gas - np.arctan(x_gas)))
gas_halo = PseudoIsothermalProfile(rho_0=gas_rho0, r_c=5.0, r_max=AND_GAS_HALO_RADIUS_KPC)

r_eval_rc = np.linspace(1, 200, 200)
M_gal_enc = (smbh.M_enc(r_eval_rc) + bulge.M_enc(r_eval_rc) + disk.M_enc(r_eval_rc)
			 + stellar_halo.M_enc(r_eval_rc) + gas_halo.M_enc(r_eval_rc))
v_gal = np.sqrt(G_kpc_Msun_Gyr * M_gal_enc / r_eval_rc) * kpc_per_Gyr_to_kms


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
R_eq = history_jeans['R_eq']
shell_mass = history_jeans['shell_mass']
rho_bg = config.rho_bg

# Radial bins for density profiles
r_bins = np.linspace(0, config.r_domain, 101)

# Select time snapshots to display
n_snaps = len(history_jeans['times'])
snap_indices = [0, n_snaps // 4, n_snaps // 2, 3 * n_snaps // 4, n_snaps - 1]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor('#111111')
fig.suptitle('1D Spherical Cavity Simulation', fontsize=16, fontweight='bold', color='white')
for ax in axes.ravel():
	ax.set_facecolor('#0d0d0d')
	ax.tick_params(colors='#aaaaaa')
	ax.xaxis.label.set_color('#cccccc')
	ax.yaxis.label.set_color('#cccccc')
	ax.title.set_color('#ffffff')
	for spine in ax.spines.values():
		spine.set_edgecolor('#333333')

# ─── Panel 1: Density profiles over time (Jeans swindle) ─────────────────
ax = axes[0, 0]
colors = plt.cm.viridis(np.linspace(0, 1, len(snap_indices)))
for i, si in enumerate(snap_indices):
	r_c, rho = measure_density_profile(history_jeans['radii'][si], shell_mass, r_bins)
	t = history_jeans['times'][si]
	ax.plot(r_c, rho / rho_bg, color=colors[i], label=f't={t:.2f} Gyr')
ax.axhline(1.0, color='#888888', ls='--', alpha=0.6, label='ρ_bg')
ax.axvline(R_eq, color='#ff4444', ls=':', alpha=0.7, label=f'R_eq={R_eq:.0f} kpc')
ax.set_xlabel('r [kpc]')
ax.set_ylabel('ρ / ρ_bg')
ax.set_title('Density Evolution (Jeans swindle)')
ax.legend(fontsize=8, facecolor='#222222', labelcolor='white', edgecolor='#555555')
ax.set_ylim(-0.1, 3.0)

# ─── Panel 2: Density profiles over time (raw forces) ────────────────────
ax = axes[0, 1]
for i, si in enumerate(snap_indices):
	r_c, rho = measure_density_profile(history_raw['radii'][si], shell_mass, r_bins)
	t = history_raw['times'][si]
	ax.plot(r_c, rho / rho_bg, color=colors[i], label=f't={t:.2f} Gyr')
ax.axhline(1.0, color='#888888', ls='--', alpha=0.6, label='ρ_bg')
ax.axvline(R_eq, color='#ff4444', ls=':', alpha=0.7, label=f'R_eq={R_eq:.0f} kpc')
ax.set_xlabel('r [kpc]')
ax.set_ylabel('ρ / ρ_bg')
ax.set_title('Density Evolution (raw forces, finite sphere)')
ax.legend(fontsize=8, facecolor='#222222', labelcolor='white', edgecolor='#555555')
ax.set_ylim(-0.1, 3.0)

# ─── Panel 3: Density profiles (warm, Jeans swindle) ─────────────────────
ax = axes[0, 2]
for i, si in enumerate(snap_indices):
	r_c, rho = measure_density_profile(history_warm['radii'][si], shell_mass, r_bins)
	t = history_warm['times'][si]
	ax.plot(r_c, rho / rho_bg, color=colors[i], label=f't={t:.2f} Gyr')
ax.axhline(1.0, color='#888888', ls='--', alpha=0.6, label='ρ_bg')
ax.axvline(R_eq, color='#ff4444', ls=':', alpha=0.7, label=f'R_eq={R_eq:.0f} kpc')
ax.set_xlabel('r [kpc]')
ax.set_ylabel('ρ / ρ_bg')
ax.set_title('Density Evolution (warm, σ=100 km/s)')
ax.legend(fontsize=8, facecolor='#222222', labelcolor='white', edgecolor='#555555')
ax.set_ylim(-0.1, 3.0)

# ─── Panel 4: Final density profile comparison (all 3 models) ────────────
ax = axes[1, 0]
final_idx = -1
for hist, label, ls in [(history_jeans, 'Jeans swindle (cold)', '-'),
						 (history_raw, 'Raw forces (finite)', '--'),
						 (history_warm, 'Jeans swindle (warm)', ':')]:
	r_c, rho = measure_density_profile(hist['radii'][final_idx], shell_mass, r_bins)
	ax.plot(r_c, rho / rho_bg, ls=ls, lw=2, label=label)
ax.axhline(1.0, color='#888888', ls='--', alpha=0.6)
ax.axvline(R_eq, color='#ff4444', ls=':', alpha=0.7, label=f'R_eq={R_eq:.0f} kpc')
ax.set_xlabel('r [kpc]')
ax.set_ylabel('ρ / ρ_bg')
ax.set_title(f'Final Density Profiles (t={history_jeans["times"][-1]:.1f} Gyr)')
ax.legend(fontsize=8, facecolor='#222222', labelcolor='white', edgecolor='#555555')
ax.set_ylim(-0.1, 3.0)

# ─── Panel 5: Enclosed deficit mass vs radius ────────────────────────────
ax = axes[1, 1]
r_eval = np.linspace(1, config.r_domain, 200)
for hist, label, ls in [(history_jeans, 'Jeans swindle (cold)', '-'),
						 (history_raw, 'Raw forces (finite)', '--'),
						 (history_warm, 'Jeans swindle (warm)', ':')]:
	_, _, M_DM = compute_rotation_curve_contribution(
		hist['radii'][-1], shell_mass, rho_bg, r_eval)
	ax.plot(r_eval, M_DM, ls=ls, lw=2, label=label)
ax.axhline(config.galaxy_mass, color='#ffaa44', ls='--', alpha=0.8, label=f'M_gal={config.galaxy_mass:.1e} M☉')
ax.axvline(R_eq, color='#ff4444', ls=':', alpha=0.7)
ax.set_xlabel('r [kpc]')
ax.set_ylabel('M_DM(r) [M☉]')
ax.set_title('Enclosed Deficit Mass (effective DM)')
ax.legend(fontsize=8, facecolor='#222222', labelcolor='white', edgecolor='#555555')
ax.set_yscale('log')

# ─── Panel 6: Rotation curve contribution ────────────────────────────────
ax = axes[1, 2]
ax.plot(r_eval, v_gal, '--', lw=1.5, color='#aaaaaa', alpha=0.8, label='Galaxy only')

# NFW reference (c=10, M200=1.3e12) — use NFWProfile class for correct r200
M200 = 1.3e12
c = 10.0
nfw = NFWProfile(M_200=M200, c=c)
M_nfw_enc = nfw.M_enc(r_eval)
v_nfw = np.sqrt(G_kpc_Msun_Gyr * (M_gal_enc + M_nfw_enc) / r_eval) * kpc_per_Gyr_to_kms
ax.plot(r_eval, v_nfw, 'b-', lw=2, alpha=0.7, label='Galaxy + NFW halo')

for hist, label, ls, color in [
		(history_jeans, 'Galaxy + cavity (cold)', '-', 'tab:green'),
		(history_warm, 'Galaxy + cavity (warm)', ':', 'tab:purple')]:
	_, v_dm, _ = compute_rotation_curve_contribution(
		hist['radii'][-1], shell_mass, rho_bg, r_eval)
	v_total = np.sqrt(v_gal**2 + v_dm**2)  # quadrature sum
	ax.plot(r_eval, v_total, ls=ls, lw=2, color=color, label=label)

ax.set_xlabel('r [kpc]')
ax.set_ylabel('v_c [km/s]')
ax.set_title('Rotation Curves')
ax.legend(fontsize=8, facecolor='#222222', labelcolor='white', edgecolor='#555555')
ax.set_ylim(0, 350)

plt.tight_layout()
plt.savefig('cavity_1d_simulation.png', dpi=150, bbox_inches='tight', facecolor='#111111')
plt.show()
print(f"\nPlot saved to cavity_1d_simulation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
for hist, name in [(history_jeans, 'Jeans swindle (cold)'),
					(history_raw, 'Raw forces (finite)'),
					(history_warm, 'Jeans swindle (warm)')]:
	final_radii = hist['radii'][-1]
	n_inside_Req = np.sum(final_radii < R_eq)
	n_inside_50 = np.sum(final_radii < 50)
	_, _, M_DM = compute_rotation_curve_contribution(
		final_radii, shell_mass, rho_bg, np.array([R_eq, 50.0, 100.0, 200.0]))
	print(f"\n  {name}:")
	print(f"    Shells inside R_eq ({R_eq:.0f} kpc): {n_inside_Req} / {config.n_shells}")
	print(f"    Shells inside 50 kpc: {n_inside_50} / {config.n_shells}")
	print(f"    M_DM at R_eq:  {M_DM[0]:.2e} M☉ (ratio to M_gal: {M_DM[0]/config.galaxy_mass:.2f})")
	print(f"    M_DM at 50 kpc:  {M_DM[1]:.2e} M☉")
	print(f"    M_DM at 100 kpc: {M_DM[2]:.2e} M☉")
	print(f"    M_DM at 200 kpc: {M_DM[3]:.2e} M☉")
