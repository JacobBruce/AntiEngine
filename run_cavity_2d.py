"""
Entry point for the 2D cavity self-limiting verification simulation.

Run from the project root:
	python run_cavity_2d.py

Tests how the cavity radius depends on the surrounding neg-mass density.
If the self-limiting mechanism holds, the cavity shrinks with increasing ρ_neg
so that the effective mass (background deficit) stays ≈ M_gal. In 2D:
  R_cavity ∝ √(M_gal / ρ_bg) for a fully evacuated cavity.

Modes:
  --animate : show animated particle visualization (default)
  --sweep   : run a neg-mass density sweep to measure cavity radius vs N_neg
"""

import argparse
import numpy as np

from simulations.cavity2d.sim import Cavity2DConfig, run_cavity_2d, measure_cavity_radius
from simulations.cavity2d.visualize import (
	animate_simulation, plot_density_profile,
	plot_averaged_profile, plot_profile_comparison,
)


def run_single(args):
	"""Run a single simulation and visualize it."""
	config = Cavity2DConfig(
		n_positive=1,
		n_negative=args.n_neg,
		mass_pos=args.mass_pos,
		mass_neg=-1.0,
		boundary_radius=args.radius,
		G=1.0,
		softening=0.5,
		dt=0.005,
		n_steps=args.steps,
		record_every=50,
		seed=42,
		vel_scale=0.0,
		restitution=1.0,
		pin_positive=True,
	)

	history = run_cavity_2d(config)

	# Show density profile
	plot_density_profile(history, snapshot_idx=-1)

	# Animate
	ani = animate_simulation(history, interval=30, steps_per_frame=2)
	return history


def run_profile(args):
	"""
	Extract and plot averaged density profiles.

	Runs multiple simulations with different N_neg values (or a single run)
	and compares the time-averaged radial density profiles.
	"""
	n_neg_values = [256, 512, 1024]
	if args.n_neg_list:
		n_neg_values = [int(x) for x in args.n_neg_list.split(',')]

	histories = []
	labels = []

	for n_neg in n_neg_values:
		print(f"\n{'='*60}")
		print(f"  Profile extraction: N_neg = {n_neg}")
		print(f"{'='*60}")

		config = Cavity2DConfig(
			n_positive=1,
			n_negative=n_neg,
			mass_pos=args.mass_pos,
			mass_neg=-1.0,
			boundary_radius=args.radius,
			G=1.0,
			softening=0.5,
			dt=0.005,
			n_steps=args.steps,
			record_every=50,
			seed=42,
			vel_scale=0.0,
			restitution=1.0,
			pin_positive=True,
		)

		history = run_cavity_2d(config)
		histories.append(history)

		rho_bg = n_neg / (np.pi * config.boundary_radius**2)
		labels.append(f"N={n_neg}, ρ={rho_bg:.3f}")

	# Individual averaged profile for first run
	plot_averaged_profile(histories[0], n_bins=80, equil_fraction=0.5)

	# Comparison across all runs
	if len(histories) > 1:
		plot_profile_comparison(histories, labels=labels, n_bins=80, equil_fraction=0.5)

	return histories


def run_sweep(args):
	"""
	Sweep over neg-mass particle counts (varying background density)
	with fixed positive mass. Measure the equilibrium cavity radius.

	If self-limiting works, the cavity should shrink with increasing ρ_bg
	so that the effective mass (deficit) stays ≈ M_pos.
	"""
	import matplotlib.pyplot as plt

	n_neg_values = [100, 200, 400, 800, 1600]
	if args.n_neg_list:
		n_neg_values = [int(x) for x in args.n_neg_list.split(',')]

	results = []

	for n_neg in n_neg_values:
		print(f"\n{'='*60}")
		print(f"  Running with N_neg = {n_neg}")
		print(f"{'='*60}")

		config = Cavity2DConfig(
			n_positive=1,
			n_negative=n_neg,
			mass_pos=args.mass_pos,
			mass_neg=-1.0,
			boundary_radius=args.radius,
			G=1.0,
			softening=0.5,
			dt=0.005,
			n_steps=args.steps,
			record_every=100,
			seed=42,
			vel_scale=0.0,
			restitution=0.8,
			pin_positive=True,
		)

		history = run_cavity_2d(config)

		# Measure cavity radius from the last 20% of frames (equilibrium)
		times, cavity_r = measure_cavity_radius(history, method='percentile', percentile=5)
		n_equil = max(1, len(cavity_r) // 5)
		equil_radius = np.mean(cavity_r[-n_equil:])
		equil_std = np.std(cavity_r[-n_equil:])

		# Background density (particles per unit area)
		area = np.pi * config.boundary_radius**2
		rho_bg = n_neg / area

		results.append({
			'n_neg': n_neg,
			'rho_bg': rho_bg,
			'cavity_radius': equil_radius,
			'cavity_std': equil_std,
			'history': history,
		})

		print(f"  Equilibrium cavity radius: {equil_radius:.2f} ± {equil_std:.2f}")
		print(f"  Background density: {rho_bg:.2f} particles/area")

	# Summary
	print(f"\n{'='*60}")
	print(f"  SWEEP RESULTS (M_pos = {args.mass_pos})")
	print(f"{'='*60}")
	print(f"  {'N_neg':>6s}  {'ρ_bg':>8s}  {'R_cavity':>10s}  {'±σ':>8s}")
	print(f"  {'-'*38}")
	for r in results:
		print(f"  {r['n_neg']:6d}  {r['rho_bg']:8.2f}  {r['cavity_radius']:10.2f}  {r['cavity_std']:8.2f}")

	# Plot: cavity radius vs background density
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
	fig.patch.set_facecolor('#111111')
	fig.suptitle(f'Cavity Radius vs Background Density (M_pos = {args.mass_pos})',
				 color='white', fontsize=14)

	for ax in (ax1, ax2):
		ax.set_facecolor('#111111')
		ax.tick_params(colors='#cccccc')
		ax.xaxis.label.set_color('#cccccc')
		ax.yaxis.label.set_color('#cccccc')
		ax.title.set_color('#ffffff')
		for spine in ax.spines.values():
			spine.set_edgecolor('#444444')

	n_neg_arr = np.array([r['n_neg'] for r in results])
	rho_arr = np.array([r['rho_bg'] for r in results])
	r_arr = np.array([r['cavity_radius'] for r in results])
	r_std = np.array([r['cavity_std'] for r in results])

	# Panel 1: cavity radius vs N_neg
	ax1.errorbar(n_neg_arr, r_arr, yerr=r_std, fmt='o-', color='#ffff44',
				 ecolor='#ffff44', capsize=4, markersize=8)
	ax1.set_xlabel('N_neg (number of negative particles)')
	ax1.set_ylabel('Equilibrium cavity radius')
	ax1.set_title('Cavity Radius vs N_neg')

	# Panel 2: cavity radius vs background density
	ax2.errorbar(rho_arr, r_arr, yerr=r_std, fmt='s-', color='#44ff88',
				 ecolor='#44ff88', capsize=4, markersize=8)
	ax2.set_xlabel('Background density (particles / area)')
	ax2.set_ylabel('Equilibrium cavity radius')
	ax2.set_title('Cavity Radius vs ρ_bg')

	# If 2D self-limiting: R_cavity ∝ 1/sqrt(ρ_bg), plot the prediction
	if len(r_arr) > 1:
		# Fit power law to data
		mask = r_arr < 0.95 * args.radius
		if np.sum(mask) >= 2:
			log_rho = np.log(rho_arr[mask])
			log_r = np.log(r_arr[mask])
			slope, intercept = np.polyfit(log_rho, log_r, 1)
			rho_fit = np.linspace(rho_arr[mask].min() * 0.8, rho_arr[mask].max() * 1.2, 50)
			r_fit = np.exp(intercept) * rho_fit**slope
			ax2.plot(rho_fit, r_fit, color='#888888', ls='--', lw=1,
					 label=f'fit: R ∝ ρ^{slope:.2f}')
		for ax in (ax1, ax2):
			ax.legend(facecolor='#222222', labelcolor='white',
					  framealpha=0.7, edgecolor='#555555')

	plt.tight_layout()
	plt.show()

	return results


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='2D Cavity Self-Limiting Test')
	parser.add_argument('--mode', choices=['animate', 'sweep', 'profile'], default='animate',
						help='Run mode: animate, sweep over densities, or extract profiles')
	parser.add_argument('--n-neg', type=int, default=500,
						help='Number of negative particles (animate mode)')
	parser.add_argument('--mass-pos', type=float, default=10.0,
						help='Positive particle mass')
	parser.add_argument('--radius', type=float, default=20.0,
						help='Boundary radius')
	parser.add_argument('--steps', type=int, default=20000,
						help='Number of integration steps')
	parser.add_argument('--n-neg-list', type=str, default=None,
						help='Comma-separated N_neg values for sweep mode')
	args = parser.parse_args()

	if args.mode == 'animate':
		run_single(args)
	elif args.mode == 'sweep':
		run_sweep(args)
	elif args.mode == 'profile':
		run_profile(args)
