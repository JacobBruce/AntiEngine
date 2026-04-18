"""
Empirical cavity profile extraction: vary M_pos, N_neg, R independently.
For each run, extract time-averaged density profile and rescale by R_bnd and rho_bg.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulations.cavity2d.sim import (
	Cavity2DConfig, run_cavity_2d, measure_density_profile, measure_effective_mass
)

# Define parameter grid: each row varies ONE parameter while holding others fixed
# Format: (label, M_pos, N_neg, R_boundary)
configs = [
	# Vary M_pos (ρ_bg fixed via N_neg/R² = 512/(π×40²))
	("M=64",    64,  512, 40),
	("M=128",  128,  512, 40),
	("M=256",  256,  512, 40),
	("M=512",  512,  512, 40),

	# Vary ρ_bg (M_pos=512 fixed, R=40 fixed, change N_neg)
	("N=128",  512,  128,  40),
	("N=256",  512,  256,  40),
	("N=512",  512,  512,  40),
	("N=1024", 512,  1024, 40),

	# Vary R (M_pos=512, N_neg=512, so ρ_bg changes with R)
	("R=30",   512,  512, 30),
	("R=50",   512,  512, 50),
	("R=70",   512,  512, 70),
]

results = []
for label, M_pos, N_neg, R_bnd in configs:
	rho_bg = N_neg / (np.pi * R_bnd**2)
	R_eq = np.sqrt(M_pos / (np.pi * rho_bg))  # fully-evacuated equilibrium radius
	print(f"\n--- {label}: M_pos={M_pos}, N_neg={N_neg}, R={R_bnd}, "
		  f"rho_bg={rho_bg:.4f}, R_eq={R_eq:.1f} ---")

	config = Cavity2DConfig(
		n_positive=1, n_negative=N_neg,
		mass_pos=M_pos, mass_neg=-1.0,
		boundary_radius=R_bnd,
		G=1.0, softening=0.5, dt=0.005,
		n_steps=15000, record_every=50,
		seed=42, vel_scale=5.0,
		restitution=1.0, pin_positive=True,
	)
	history = run_cavity_2d(config)
	r, density, density_std, rho = measure_density_profile(history, n_bins=100, equil_fraction=0.5)
	r_eval, M_eff, M_eff_std = measure_effective_mass(history, n_bins=100, equil_fraction=0.5)

	# Measure ρ_bg from the outermost bins (boundary density = true background).
	# In an infinite universe ρ_bg is the undisturbed density far from the galaxy.
	# In our finite box, the boundary is the furthest point from the galaxy, so
	# the density there is the best estimate of the true background.
	# Use the last 2% of radial bins (last 2 of 100 bins).
	outer_frac = 0.02
	outer_mask = r > (1.0 - outer_frac) * R_bnd
	if outer_mask.sum() >= 2:
		rho_bg_boundary = np.mean(density[outer_mask])
	else:
		rho_bg_boundary = density[-1]

	# Also record the domain-average for comparison
	rho_bg_domain = rho_bg  # N/(πR²)

	results.append({
		'label': label, 'M_pos': M_pos, 'N_neg': N_neg, 'R_bnd': R_bnd,
		'rho_bg': rho_bg_boundary, 'rho_bg_domain': rho_bg_domain, 'R_eq': R_eq,
		'r': r, 'density': density, 'density_std': density_std,
		'r_eval': r_eval, 'M_eff': M_eff, 'M_eff_std': M_eff_std,
	})
	peak_meff = np.max(M_eff)
	r_peak = r_eval[np.argmax(M_eff)]
	print(f"  Peak M_eff = {peak_meff:.1f} (at r={r_peak:.1f}), "
		  f"M_eff/M_pos = {peak_meff/M_pos:.2f}")
	print(f"  ρ_bg boundary = {rho_bg_boundary:.4f}, "
		  f"ρ_bg domain avg = {rho_bg_domain:.4f}, "
		  f"ratio = {rho_bg_boundary/rho_bg_domain:.2f}")

np.save('images/cavity_profile_results.npy', results, allow_pickle=True)
print("\nDone — saved results to images/cavity_profile_results.npy")

# ============================================================
# Plot 1: Raw profiles grouped by which parameter varies
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor('#111111')
groups = [
	("Vary M_pos (ρ_bg≈0.10)", results[0:4]),
	("Vary N_neg (M_pos=128)", results[4:8]),
	("Vary R_bnd (M_pos=128, N=512)", results[8:11]),
]
colors = ['#ff4444', '#ff8844', '#44ff44', '#4488ff', '#ff44ff']

for ax, (title, group) in zip(axes, groups):
	ax.set_facecolor('#0d0d0d')
	ax.tick_params(colors='#aaaaaa')
	ax.xaxis.label.set_color('#cccccc')
	ax.yaxis.label.set_color('#cccccc')
	ax.title.set_color('#ffffff')
	for spine in ax.spines.values():
		spine.set_edgecolor('#333333')

	for i, res in enumerate(group):
		x = res['r'] / res['R_bnd']
		y = res['density'] / res['rho_bg']
		ax.plot(x, y, color=colors[i], lw=1.5, label=res['label'])
	ax.axhline(1.0, color='#555555', ls='--', lw=0.8)
	ax.axvline(1.0, color='#555555', ls=':', lw=0.8, alpha=0.5)
	ax.set_xlabel('r / R_bnd')
	ax.set_ylabel('ρ / ρ_bg')
	ax.set_title(title)
	ax.set_xlim(0, 1.05)
	ax.set_ylim(-0.05, 3.5)
	ax.legend(facecolor='#222222', labelcolor='white', edgecolor='#555555', fontsize=9)

plt.tight_layout()
plt.savefig('images/cavity_profiles_raw.png', dpi=120, facecolor='#111111')
print("Saved cavity_profiles_raw.png")

# ============================================================
# Plot 2: Rescaled profiles with r / R_bnd
# Recovery function: ρ_neg/ρ_bg as a function of r/R_bnd
# Normalized fit functions constrained to f(0)=0, f(1)=1
# ============================================================
from scipy.optimize import curve_fit
from scipy.special import betainc as sp_betainc

# Fit functions for the recovery profile ρ_neg/ρ_bg = f(x), x = r/R_bnd
# All functions are constrained: f(0) = 0, f(1) = 1

def norm_logistic(x, a, s):
	"""Logistic normalized to f(1)=1: f(x) = (s^a+1) × x^a / (s^a + x^a)"""
	s_a = np.power(s, a)
	return (s_a + 1.0) * np.power(x, a) / (s_a + np.power(x, a))

def norm_stretched_exp(x, s, b):
	"""Stretched exp normalized to f(1)=1: f(x) = (1-exp(-(x/s)^b)) / (1-exp(-(1/s)^b))"""
	norm = 1.0 - np.exp(-np.power(1.0 / s, b))
	return (1.0 - np.exp(-np.power(x / s, b))) / norm

def beta_cdf(x, a, b):
	"""Regularized incomplete beta function I_x(a, b). Goes 0→1 on [0,1]."""
	x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
	return sp_betainc(a, b, x)

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.patch.set_facecolor('#111111')

for ax in axes2.ravel():
	ax.set_facecolor('#0d0d0d')
	ax.tick_params(colors='#aaaaaa')
	ax.xaxis.label.set_color('#cccccc')
	ax.yaxis.label.set_color('#cccccc')
	ax.title.set_color('#ffffff')
	for spine in ax.spines.values():
		spine.set_edgecolor('#333333')

all_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(results)))

# Top row: density profiles rescaled by R_bnd
ax_lin, ax_log, ax_meff = axes2[0]
# Bottom row: per-group fits
ax_fit_m, ax_fit_n, ax_fit_r = axes2[1]

print("\n=== Profile fits (x = r/R_bnd, normalized f(0)=0, f(1)=1) ===")
print(f"{'Run':<10} {'R_bnd/R_eq':>10} {'ρ_bnd/ρ_avg':>11} "
	  f"{'NLog α':>8} {'NLog s':>8} {'RMS_nl':>8} "
	  f"{'NSE s':>8} {'NSE b':>8} {'RMS_nse':>8} "
	  f"{'Beta a':>8} {'Beta b':>8} {'RMS_beta':>8}")

fit_results = []
for i, res in enumerate(results):
	x = res['r'] / res['R_bnd']
	y = res['density'] / res['rho_bg']

	# Plot full profile
	ax_lin.plot(x, y, color=all_colors[i], lw=1.0, alpha=0.8, label=res['label'])
	ax_log.semilogy(x, np.clip(y, 1e-6, None), color=all_colors[i],
					lw=1.0, alpha=0.8, label=res['label'])

	# M_eff rescaled
	x_m = res['r_eval'] / res['R_bnd']
	ax_meff.plot(x_m, res['M_eff'] / res['M_pos'],
				 color=all_colors[i], lw=1.0, alpha=0.8, label=res['label'])

	# Fit region: full profile (exclude origin bin)
	mask = (x > 0.01) & (x < 0.999) & (y >= 0)
	x_fit = x[mask]
	y_fit = y[mask]

	ratio = res['R_bnd'] / res['R_eq']

	# Fit normalized logistic
	try:
		p_nlog, _ = curve_fit(norm_logistic, x_fit, y_fit,
							  p0=[3.0, 0.5], bounds=([0.5, 0.01], [20.0, 2.0]))
		rms_nlog = np.sqrt(np.mean((norm_logistic(x_fit, *p_nlog) - y_fit)**2))
	except Exception:
		p_nlog = [np.nan, np.nan]; rms_nlog = np.nan

	# Fit normalized stretched exponential
	try:
		p_nse, _ = curve_fit(norm_stretched_exp, x_fit, y_fit,
							 p0=[0.5, 2.0], bounds=([0.05, 0.5], [2.0, 10.0]))
		rms_nse = np.sqrt(np.mean((norm_stretched_exp(x_fit, *p_nse) - y_fit)**2))
	except Exception:
		p_nse = [np.nan, np.nan]; rms_nse = np.nan

	# Fit beta CDF
	try:
		p_beta, _ = curve_fit(beta_cdf, x_fit, y_fit,
							  p0=[2.0, 1.5], bounds=([0.1, 0.1], [20.0, 20.0]))
		rms_beta = np.sqrt(np.mean((beta_cdf(x_fit, *p_beta) - y_fit)**2))
	except Exception:
		p_beta = [np.nan, np.nan]; rms_beta = np.nan

	fit_results.append({
		'label': res['label'], 'ratio': ratio,
		'norm_logistic': p_nlog, 'rms_nlog': rms_nlog,
		'norm_strexp': p_nse, 'rms_nse': rms_nse,
		'beta': p_beta, 'rms_beta': rms_beta,
	})

	bg_ratio = res['rho_bg'] / res['rho_bg_domain'] if res.get('rho_bg_domain', 0) > 0 else 0

	print(f"{res['label']:<10} {ratio:>10.2f} {bg_ratio:>11.2f} "
		  f"{p_nlog[0]:>8.2f} {p_nlog[1]:>8.3f} {rms_nlog:>8.4f} "
		  f"{p_nse[0]:>8.3f} {p_nse[1]:>8.2f} {rms_nse:>8.4f} "
		  f"{p_beta[0]:>8.2f} {p_beta[1]:>8.2f} {rms_beta:>8.4f}")

# Configure top-row axes
ax_lin.axhline(1.0, color='#555555', ls='--', lw=0.8)
ax_lin.set_xlabel('r / R_bnd')
ax_lin.set_ylabel('ρ / ρ_bg')
ax_lin.set_title('Recovery profile (linear)')
ax_lin.set_xlim(0, 1.05)
ax_lin.set_ylim(-0.05, 1.3)
ax_lin.legend(facecolor='#222222', labelcolor='white', edgecolor='#555555',
			  fontsize=6, ncol=2)

ax_log.axhline(1.0, color='#555555', ls='--', lw=0.8)
ax_log.set_xlabel('r / R_bnd')
ax_log.set_ylabel('ρ / ρ_bg')
ax_log.set_title('Recovery profile (log)')
ax_log.set_xlim(0, 1.05)
ax_log.legend(facecolor='#222222', labelcolor='white', edgecolor='#555555',
			  fontsize=6, ncol=2)

ax_meff.axhline(1.0, color='#888888', ls='--', lw=0.8, label='M_eff = M_pos')
ax_meff.set_xlabel('r / R_bnd')
ax_meff.set_ylabel('M_eff / M_pos')
ax_meff.set_title('Effective mass (deficit)')
ax_meff.set_xlim(0, 1.05)
ax_meff.legend(facecolor='#222222', labelcolor='white', edgecolor='#555555',
			   fontsize=6, ncol=2)

# Bottom row: best fits per group (show all 3 forms for comparison)
fit_groups = [
	("Vary M_pos", results[0:4], fit_results[0:4], colors),
	("Vary N_neg", results[4:8], fit_results[4:8], colors),
	("Vary R_bnd", results[8:11], fit_results[8:11], colors),
]
x_plot = np.linspace(0.001, 1.0, 200)

for ax_fit, (title, group, fits, cols) in zip([ax_fit_m, ax_fit_n, ax_fit_r], fit_groups):
	for j, (res, fr) in enumerate(zip(group, fits)):
		x = res['r'] / res['R_bnd']
		y = res['density'] / res['rho_bg']
		ax_fit.plot(x, y, color=cols[j], lw=1.5, alpha=0.8)

		# Pick the best fit (lowest RMS) from the three forms
		rms_vals = [
			('nlog', fr.get('rms_nlog', np.nan), fr.get('norm_logistic', [np.nan]*2)),
			('nse', fr.get('rms_nse', np.nan), fr.get('norm_strexp', [np.nan]*2)),
			('beta', fr.get('rms_beta', np.nan), fr.get('beta', [np.nan]*2)),
		]
		best = min(rms_vals, key=lambda t: t[1] if not np.isnan(t[1]) else 999)
		name, rms, params = best

		if name == 'nlog' and not np.isnan(rms):
			ax_fit.plot(x_plot, norm_logistic(x_plot, *params),
						color=cols[j], ls='--', lw=1.0, alpha=0.6)
			lbl = f"{res['label']} nlog α={params[0]:.1f} s={params[1]:.2f} rms={rms:.3f}"
		elif name == 'nse' and not np.isnan(rms):
			ax_fit.plot(x_plot, norm_stretched_exp(x_plot, *params),
						color=cols[j], ls='--', lw=1.0, alpha=0.6)
			lbl = f"{res['label']} nse s={params[0]:.2f} b={params[1]:.1f} rms={rms:.3f}"
		elif name == 'beta' and not np.isnan(rms):
			ax_fit.plot(x_plot, beta_cdf(x_plot, *params),
						color=cols[j], ls='--', lw=1.0, alpha=0.6)
			lbl = f"{res['label']} beta a={params[0]:.1f} b={params[1]:.1f} rms={rms:.3f}"
		else:
			lbl = f"{res['label']} (fit failed)"
		ax_fit.plot([], [], color=cols[j], ls='-', lw=1.5, label=lbl)

	ax_fit.axhline(1.0, color='#555555', ls='--', lw=0.8)
	ax_fit.set_xlabel('r / R_bnd')
	ax_fit.set_ylabel('ρ / ρ_bg')
	ax_fit.set_title(f'{title} — best fits (full range)')
	ax_fit.set_xlim(0, 1.05)
	ax_fit.set_ylim(-0.05, None)
	ax_fit.legend(facecolor='#222222', labelcolor='white', edgecolor='#555555', fontsize=7)
	ax_fit.grid(True, alpha=0.2, color='#555555')

fig2.suptitle('Cavity Recovery Profile — rescaled by R_bnd',
			  color='white', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('images/cavity_profiles_rescaled.png', dpi=120, facecolor='#111111',
			bbox_inches='tight')
print("Saved cavity_profiles_rescaled.png")
