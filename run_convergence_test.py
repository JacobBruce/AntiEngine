"""
Convergence test for the cosmological simulation.

Runs the same physical setup at multiple resolutions to determine whether
key observables are robust to resolution changes.

Two modes:
  - Resolution convergence: vary n_per_dim (particle count) at fixed 2:1 ratio
  - Grid-ratio convergence: vary n_grid / n_per_dim at fixed particle count

Key observable depends on IC mode:
  - Anti-correlated ICs: cross-correlation at a=1 (should converge to stable value)
  - Correlated ICs: zero-crossing time (when species become uncorrelated)

Usage:
    python run_convergence_test.py              # resolution convergence
    python run_convergence_test.py --grid-ratio # grid-ratio convergence
"""

import numpy as np
import matplotlib.pyplot as plt
from simulations.cosmological.initial_conditions import CosmoConfig
from simulations.cosmological.sim import run_cosmological_simulation


# ── Resolution levels to test ────────────────────────────────────────────────
# Each entry: (n_per_dim, n_grid, n_steps)
# n_grid = 2 × n_per_dim is the standard PM ratio
RESOLUTIONS = [
	(16, 32,  1000),   # very coarse — fast baseline
	(32, 64,  1500),   # medium
	(48, 96,  1800),   # fine
	(64, 128, 2000),   # production
]

# Fixed particle count, varying n_grid / n_per_dim ratio
GRID_RATIOS = [
	(32, 32,  1500),   # 1:1 — under-resolved
	(32, 48,  1500),   # 1.5:1
	(32, 64,  1500),   # 2:1 — standard minimum
	(32, 96,  1500),   # 3:1
	(32, 128, 1500),   # 4:1
	(32, 192, 1500),   # 6:1 — over-resolved
]

# ── Default configuration ────────────────────────────────────────────────────
DEFAULT_CONFIG = CosmoConfig(
	box_size=100_000.0,
	neg_mass_ratio=1.0,
	H0=0.07159,
	rho_crit=136.0,
	a_initial=0.1,
	a_final=5.0,
	perturbation_amplitude=1.0,
	anti_correlated=True,
	spectral_index=1.0,
	seed=42,
)


def find_zero_crossing(times, values):
	"""Find the time where values crosses zero (linear interpolation)."""
	for i in range(1, len(values)):
		if values[i - 1] > 0 and values[i] <= 0:
			t0, t1 = times[i - 1], times[i]
			v0, v1 = values[i - 1], values[i]
			return t0 + (0 - v0) * (t1 - t0) / (v1 - v0)
	return None


def find_corr_at_a(scale_factors, cross_corr, a_target=1.0):
	"""Find the cross-correlation at the snapshot nearest to a_target."""
	idx = np.argmin(np.abs(np.array(scale_factors) - a_target))
	return scale_factors[idx], cross_corr[idx]


def _style_axes(ax):
	ax.set_facecolor('#0d0d0d')
	ax.tick_params(colors='#aaaaaa', labelsize=9)
	ax.xaxis.label.set_color('#cccccc')
	ax.yaxis.label.set_color('#cccccc')
	ax.title.set_color('#ffffff')
	for spine in ax.spines.values():
		spine.set_edgecolor('#333333')


def run_convergence(config=None):
	"""
	Resolution convergence test.

	Varies particle count (n_per_dim) at fixed 2:1 grid ratio.
	The config parameter sets the physics; n_per_dim/n_grid/n_steps are
	overridden per resolution level.
	"""
	if config is None:
		config = DEFAULT_CONFIG
	anti_corr = config.anti_correlated

	results = []

	for n_per_dim, n_grid, n_steps in RESOLUTIONS:
		print(f"\n{'='*60}")
		print(f"Resolution: {n_per_dim}³ particles, {n_grid}³ grid, {n_steps} steps")
		print(f"{'='*60}")

		run_config = CosmoConfig(
			box_size=config.box_size,
			n_grid=n_grid,
			n_per_dim=n_per_dim,
			neg_mass_ratio=config.neg_mass_ratio,
			H0=config.H0,
			rho_crit=config.rho_crit,
			a_initial=config.a_initial,
			a_final=config.a_final,
			perturbation_amplitude=config.perturbation_amplitude,
			neg_perturbation_amplitude=config.neg_perturbation_amplitude,
			anti_correlated=config.anti_correlated,
			spectral_index=config.spectral_index,
			seed=config.seed,
		)

		history = run_cosmological_simulation(run_config, n_steps=n_steps, record_every=10)

		times = history['time']
		cross_corr = history['cross_correlation']
		t_zero = find_zero_crossing(times, cross_corr)
		a_1, corr_1 = find_corr_at_a(history['scale_factor'], cross_corr, 1.0)

		results.append({
			'n_per_dim': n_per_dim,
			'n_grid': n_grid,
			'times': times,
			'cross_corr': cross_corr,
			'scale_factor': history['scale_factor'],
			't_zero': t_zero,
			'corr_at_a1': corr_1,
			'a_at_a1': a_1,
		})

		if anti_corr:
			print(f"  Corr at a≈1 ({a_1:.3f}): {corr_1:.4f}")
		else:
			print(f"  Zero crossing: {t_zero:.2f} Gyr" if t_zero else "  No zero crossing")

	# ── Summary table ────────────────────────────────────────────────────────
	print(f"\n{'='*60}")
	print("Convergence summary")
	print(f"{'='*60}")

	if anti_corr:
		print(f"{'Resolution':>15s}  {'Grid':>8s}  {'Corr at a≈1':>12s}  {'a':>6s}")
		print(f"{'-'*15}  {'-'*8}  {'-'*12}  {'-'*6}")
		for r in results:
			print(f"{r['n_per_dim']:>5d}³ particles  {r['n_grid']:>5d}³  "
				  f"{r['corr_at_a1']:>12.4f}  {r['a_at_a1']:>6.3f}")
	else:
		print(f"{'Resolution':>15s}  {'Grid':>8s}  {'Zero-cross (Gyr)':>16s}  {'a at cross':>10s}")
		print(f"{'-'*15}  {'-'*8}  {'-'*16}  {'-'*10}")
		for r in results:
			t_z = r['t_zero']
			if t_z is not None:
				idx = np.searchsorted(r['times'], t_z)
				idx = min(idx, len(r['scale_factor']) - 1)
				a_z = r['scale_factor'][idx]
				print(f"{r['n_per_dim']:>5d}³ particles  {r['n_grid']:>5d}³  {t_z:>16.2f}  {a_z:>10.4f}")
			else:
				print(f"{r['n_per_dim']:>5d}³ particles  {r['n_grid']:>5d}³  {'not reached':>16s}  {'—':>10s}")

	# ── Plot ─────────────────────────────────────────────────────────────────
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#111111')
	for ax in [ax1, ax2]:
		_style_axes(ax)

	colors = ['#ff6666', '#ffaa44', '#66cc66', '#4499ff']
	for i, r in enumerate(results):
		label = f"{r['n_per_dim']}³ / {r['n_grid']}³"
		ax1.plot(r['scale_factor'], r['cross_corr'], color=colors[i % len(colors)],
				 linewidth=1.5, label=label)

	ax1.axhline(0, color='#555555', linewidth=0.5, linestyle='-')
	ax1.axvline(1.0, color='#ffffff', linewidth=1, linestyle='--', alpha=0.3,
				label='a = 1')
	ax1.set_xlabel('Scale factor a')
	ax1.set_ylabel('Cross-correlation')
	ax1.set_title('Cross-correlation convergence')
	ax1.legend(fontsize=8, labelcolor='#cccccc', facecolor='#1a1a1a',
			   edgecolor='#333333')
	ax1.set_ylim(-1.1, 1.1)

	# Observable vs resolution
	n_vals = [r['n_per_dim'] for r in results]
	if anti_corr:
		obs_vals = [r['corr_at_a1'] for r in results]
		ax2.plot(n_vals, obs_vals, 'o-', color='#4499ff', markersize=8, linewidth=2)
		ax2.set_ylabel('Cross-correlation at a ≈ 1')
		ax2.set_title('Convergence at a = 1')
	else:
		obs_vals = [r['t_zero'] for r in results if r['t_zero'] is not None]
		n_valid = [r['n_per_dim'] for r in results if r['t_zero'] is not None]
		if len(n_valid) > 0:
			ax2.plot(n_valid, obs_vals, 'o-', color='#4499ff', markersize=8, linewidth=2)
		ax2.set_ylabel('Zero-crossing time (Gyr)')
		ax2.set_title('Zero-crossing convergence')

	ax2.set_xlabel('n_per_dim (particles per dimension)')
	ax2.legend(fontsize=8, labelcolor='#cccccc', facecolor='#1a1a1a',
			   edgecolor='#333333')

	mode_str = 'anti-correlated' if anti_corr else 'correlated'
	fig.suptitle(f'Resolution convergence test ({mode_str} ICs)', color='white', fontsize=14)
	plt.tight_layout()
	plt.savefig('convergence_test.png', dpi=150, facecolor='#111111')
	print(f"\nPlot saved to convergence_test.png")
	plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Grid-ratio convergence test
# ─────────────────────────────────────────────────────────────────────────────

# Fixed particle count, varying n_grid / n_per_dim ratio
GRID_RATIOS = [
	(32, 32,  1500),   # 1:1 — under-resolved
	(32, 48,  1500),   # 1.5:1
	(32, 64,  1500),   # 2:1 — standard minimum
	(32, 96,  1500),   # 3:1
	(32, 128, 1500),   # 4:1
	(32, 192, 1500),   # 6:1 — over-resolved
]


def run_grid_ratio_convergence(config=None):
	"""
	Grid-ratio convergence test.

	Varies n_grid / n_per_dim ratio at fixed particle count.
	The config parameter sets the physics; n_per_dim/n_grid/n_steps are
	overridden per ratio level.
	"""
	if config is None:
		config = DEFAULT_CONFIG
	anti_corr = config.anti_correlated

	results = []

	for n_per_dim, n_grid, n_steps in GRID_RATIOS:
		ratio = n_grid / n_per_dim
		print(f"\n{'='*60}")
		print(f"Grid ratio test: {n_per_dim}³ particles, {n_grid}³ grid "
			  f"(ratio {ratio:.1f}:1), {n_steps} steps")
		print(f"{'='*60}")

		run_config = CosmoConfig(
			box_size=config.box_size,
			n_grid=n_grid,
			n_per_dim=n_per_dim,
			neg_mass_ratio=config.neg_mass_ratio,
			H0=config.H0,
			rho_crit=config.rho_crit,
			a_initial=config.a_initial,
			a_final=config.a_final,
			perturbation_amplitude=config.perturbation_amplitude,
			neg_perturbation_amplitude=config.neg_perturbation_amplitude,
			anti_correlated=config.anti_correlated,
			spectral_index=config.spectral_index,
			seed=config.seed,
		)

		history = run_cosmological_simulation(run_config, n_steps=n_steps, record_every=10)

		times = history['time']
		cross_corr = history['cross_correlation']
		t_zero = find_zero_crossing(times, cross_corr)
		a_1, corr_1 = find_corr_at_a(history['scale_factor'], cross_corr, 1.0)

		results.append({
			'n_per_dim': n_per_dim,
			'n_grid': n_grid,
			'ratio': ratio,
			'times': times,
			'cross_corr': cross_corr,
			'scale_factor': history['scale_factor'],
			't_zero': t_zero,
			'corr_at_a1': corr_1,
			'a_at_a1': a_1,
		})

		if anti_corr:
			print(f"  Corr at a≈1 ({a_1:.3f}): {corr_1:.4f}")
		else:
			t_str = f"{t_zero:.2f} Gyr" if t_zero else "not reached"
			print(f"  Zero crossing: {t_str}")

	# ── Summary table ────────────────────────────────────────────────────────
	print(f"\n{'='*60}")
	print("Grid-ratio convergence summary")
	print(f"{'='*60}")

	if anti_corr:
		print(f"{'Particles':>12s}  {'Grid':>8s}  {'Ratio':>6s}  "
			  f"{'Corr at a≈1':>12s}  {'a':>6s}")
		print(f"{'-'*12}  {'-'*8}  {'-'*6}  {'-'*12}  {'-'*6}")
		for r in results:
			print(f"{r['n_per_dim']:>5d}³       {r['n_grid']:>5d}³  "
				  f"{r['ratio']:>5.1f}×  {r['corr_at_a1']:>12.4f}  {r['a_at_a1']:>6.3f}")
	else:
		print(f"{'Particles':>12s}  {'Grid':>8s}  {'Ratio':>6s}  "
			  f"{'Zero-cross (Gyr)':>16s}  {'a at cross':>10s}")
		print(f"{'-'*12}  {'-'*8}  {'-'*6}  {'-'*16}  {'-'*10}")
		for r in results:
			t_z = r['t_zero']
			if t_z is not None:
				idx = np.searchsorted(r['times'], t_z)
				idx = min(idx, len(r['scale_factor']) - 1)
				a_z = r['scale_factor'][idx]
				print(f"{r['n_per_dim']:>5d}³       {r['n_grid']:>5d}³  "
					  f"{r['ratio']:>5.1f}×  {t_z:>16.2f}  {a_z:>10.4f}")
			else:
				print(f"{r['n_per_dim']:>5d}³       {r['n_grid']:>5d}³  "
					  f"{r['ratio']:>5.1f}×  {'not reached':>16s}  {'—':>10s}")

	# ── Plot ─────────────────────────────────────────────────────────────────
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#111111')
	for ax in [ax1, ax2]:
		_style_axes(ax)

	colors = ['#ff4444', '#ff8844', '#ffcc44', '#66cc66', '#4499ff', '#aa66ff']
	for i, r in enumerate(results):
		label = f"{r['ratio']:.1f}:1 ({r['n_grid']}³ grid)"
		ax1.plot(r['scale_factor'], r['cross_corr'], color=colors[i % len(colors)],
				 linewidth=1.5, label=label)

	ax1.axhline(0, color='#555555', linewidth=0.5, linestyle='-')
	ax1.axvline(1.0, color='#ffffff', linewidth=1, linestyle='--', alpha=0.3,
				label='a = 1')
	ax1.set_xlabel('Scale factor a')
	ax1.set_ylabel('Cross-correlation')
	ax1.set_title(f'Cross-correlation vs grid ratio ({GRID_RATIOS[0][0]}³ particles)')
	ax1.legend(fontsize=7, labelcolor='#cccccc', facecolor='#1a1a1a',
			   edgecolor='#333333', loc='lower left')
	ax1.set_ylim(-1.1, 1.1)

	# Observable vs ratio
	ratio_vals = [r['ratio'] for r in results]
	if anti_corr:
		obs_vals = [r['corr_at_a1'] for r in results]
		ax2.plot(ratio_vals, obs_vals, 'o-', color='#4499ff', markersize=8, linewidth=2)
		ax2.set_ylabel('Cross-correlation at a ≈ 1')
		ax2.set_title('Convergence at a = 1 vs force resolution')
	else:
		obs_vals = [r['t_zero'] for r in results if r['t_zero'] is not None]
		r_valid = [r['ratio'] for r in results if r['t_zero'] is not None]
		if len(r_valid) > 0:
			ax2.plot(r_valid, obs_vals, 'o-', color='#4499ff', markersize=8, linewidth=2)
		ax2.set_ylabel('Zero-crossing time (Gyr)')
		ax2.set_title('Zero-crossing vs force resolution')

	ax2.set_xlabel('n_grid / n_per_dim ratio')
	ax2.legend(fontsize=8, labelcolor='#cccccc', facecolor='#1a1a1a',
			   edgecolor='#333333')

	mode_str = 'anti-correlated' if anti_corr else 'correlated'
	fig.suptitle(f'Grid-ratio convergence test ({mode_str} ICs)', color='white', fontsize=14)
	plt.tight_layout()
	plt.savefig('grid_ratio_convergence.png', dpi=150, facecolor='#111111')
	print(f"\nPlot saved to grid_ratio_convergence.png")
	plt.show()


if __name__ == '__main__':
	import sys
	if '--grid-ratio' in sys.argv:
		run_grid_ratio_convergence()
	else:
		run_convergence()
		print("\nRun with --grid-ratio for the grid-ratio convergence test.")
