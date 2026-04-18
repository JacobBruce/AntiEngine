"""
Visualization for the 2D cavity self-limiting simulation.

Provides:
  - Animated scatter plot of particles inside the circular boundary
  - Cavity radius evolution over time
  - Radial density profile at the final snapshot
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

from .sim import measure_cavity_radius, measure_density_profile, measure_effective_mass


def animate_simulation(history, interval=50, steps_per_frame=1):
	"""
	Animated scatter plot of the 2D cavity simulation.

	Shows positive particles (blue) and negative particles (red) inside
	the circular boundary, with a real-time cavity radius indicator.
	"""
	config = history['config']
	positions_list = history['positions'][::steps_per_frame]
	masses = history['masses']
	times = np.array(history['time'][::steps_per_frame])

	pos_mask = masses > 0
	neg_mask = masses < 0
	n_pos = config.n_positive

	R = config.boundary_radius
	lim = R * 1.15

	# Pre-compute cavity radii for all displayed frames
	neg_radii_per_frame = []
	for pos in positions_list:
		neg_pos = pos[n_pos:]
		neg_r = np.linalg.norm(neg_pos, axis=1)
		neg_radii_per_frame.append(np.percentile(neg_r, 5))

	# Layout: particle view + cavity radius plot
	fig, (ax_sim, ax_radius) = plt.subplots(1, 2, figsize=(14, 6))
	fig.patch.set_facecolor('#111111')
	fig.suptitle('2D Cavity Self-Limiting Test', color='white', fontsize=14)

	for ax in (ax_sim, ax_radius):
		ax.set_facecolor('#111111')
		ax.tick_params(colors='#cccccc')
		ax.xaxis.label.set_color('#cccccc')
		ax.yaxis.label.set_color('#cccccc')
		ax.title.set_color('#ffffff')
		for spine in ax.spines.values():
			spine.set_edgecolor('#444444')

	# ── Particle panel ──
	ax_sim.set_xlim(-lim, lim)
	ax_sim.set_ylim(-lim, lim)
	ax_sim.set_aspect('equal')
	ax_sim.set_xlabel('x')
	ax_sim.set_ylabel('y')
	ax_sim.set_title('Particles')

	# Boundary circle
	boundary = Circle((0, 0), R, fill=False, edgecolor='#666666', linewidth=1.5, linestyle='--')
	ax_sim.add_patch(boundary)

	# Cavity radius circle (updated each frame)
	cavity_circle = Circle((0, 0), neg_radii_per_frame[0], fill=False,
						   edgecolor='#ffff44', linewidth=1.5, linestyle='-', alpha=0.8)
	ax_sim.add_patch(cavity_circle)

	scat_neg = ax_sim.scatter([], [], c='#ff4444', s=8, alpha=0.6, label='−mass')
	scat_pos = ax_sim.scatter([], [], c='#4488ff', s=40, alpha=1.0, label='+mass', zorder=5)
	time_text = ax_sim.text(
		0.02, 0.97, '', transform=ax_sim.transAxes,
		color='white', va='top', fontsize=9, family='monospace',
	)
	ax_sim.legend(
		loc='upper right', facecolor='#222222',
		labelcolor='white', framealpha=0.7, edgecolor='#555555',
	)

	# ── Cavity radius panel ──
	ax_radius.plot(times, neg_radii_per_frame, color='#ffff44', lw=1.2, alpha=0.6)
	ax_radius.set_xlabel('time')
	ax_radius.set_ylabel('cavity radius (5th pctl)')
	ax_radius.set_title('Cavity Radius')
	radius_marker = ax_radius.axvline(times[0], color='#ffff44', lw=1.2, alpha=0.8)

	def update(frame_idx):
		pos = positions_list[frame_idx]
		scat_pos.set_offsets(pos[pos_mask])
		scat_neg.set_offsets(pos[neg_mask])
		time_text.set_text(f't = {times[frame_idx]:.2f}')
		cavity_circle.set_radius(neg_radii_per_frame[frame_idx])
		radius_marker.set_xdata([times[frame_idx]])
		return scat_pos, scat_neg, time_text, cavity_circle, radius_marker

	ani = animation.FuncAnimation(
		fig, update,
		frames=len(positions_list),
		interval=interval,
		blit=False,  # Circle patch doesn't work well with blit
	)

	plt.tight_layout()
	plt.show()
	return ani


def plot_density_profile(history, snapshot_idx=-1):
	"""
	Plot the radial density profile of neg-mass particles at a given snapshot.

	Shows the density as a function of radius, with the cavity visible as
	the depleted inner region.
	"""
	config = history['config']
	n_pos = config.n_positive
	R = config.boundary_radius

	pos = history['positions'][snapshot_idx]
	neg_pos = pos[n_pos:]
	neg_r = np.linalg.norm(neg_pos, axis=1)

	# Radial bins
	r_bins = np.linspace(0, R, 60)
	r_centres = 0.5 * (r_bins[:-1] + r_bins[1:])
	areas = np.pi * (r_bins[1:]**2 - r_bins[:-1]**2)
	counts, _ = np.histogram(neg_r, bins=r_bins)
	density = counts / areas  # particles per unit area

	# Mean density
	mean_density = config.n_negative / (np.pi * R**2)

	fig, ax = plt.subplots(figsize=(8, 5))
	fig.patch.set_facecolor('#111111')
	ax.set_facecolor('#111111')
	ax.tick_params(colors='#cccccc')
	ax.xaxis.label.set_color('#cccccc')
	ax.yaxis.label.set_color('#cccccc')
	ax.title.set_color('#ffffff')
	for spine in ax.spines.values():
		spine.set_edgecolor('#444444')

	ax.bar(r_centres, density, width=np.diff(r_bins), color='#ff4444', alpha=0.7, edgecolor='#cc2222')
	ax.axhline(mean_density, color='#888888', ls='--', lw=1, label=f'mean = {mean_density:.3f}')

	# Mark cavity radius
	cavity_r = np.percentile(neg_r, 5)
	ax.axvline(cavity_r, color='#ffff44', ls='-', lw=1.5, label=f'cavity r = {cavity_r:.2f}')

	ax.set_xlabel('radius')
	ax.set_ylabel('surface density (particles / area)')
	ax.set_title(f'Neg-Mass Radial Density Profile (t={history["time"][snapshot_idx]:.2f})')
	ax.legend(facecolor='#222222', labelcolor='white', framealpha=0.7, edgecolor='#555555')

	plt.tight_layout()
	plt.show()
	return fig


def _style_ax(ax):
	"""Apply dark theme to an axis."""
	ax.set_facecolor('#111111')
	ax.tick_params(colors='#cccccc')
	ax.xaxis.label.set_color('#cccccc')
	ax.yaxis.label.set_color('#cccccc')
	ax.title.set_color('#ffffff')
	for spine in ax.spines.values():
		spine.set_edgecolor('#444444')


def plot_averaged_profile(history, n_bins=80, equil_fraction=0.5):
	"""
	Plot the time-averaged radial density profile and effective mass curve.

	3-panel plot:
	  1. Surface density ρ(r) vs r (log scale)
	  2. Normalized density ρ(r)/ρ_bg vs r
	  3. Effective (deficit) mass M_eff(r) vs r
	"""
	config = history['config']

	r_centres, density, density_std, rho_bg = measure_density_profile(
		history, n_bins=n_bins, equil_fraction=equil_fraction)
	r_mass, M_eff, M_eff_std = measure_effective_mass(
		history, n_bins=n_bins, equil_fraction=equil_fraction)

	fig, axes = plt.subplots(1, 3, figsize=(18, 5))
	fig.patch.set_facecolor('#111111')
	fig.suptitle(
		f'Cavity Density Profile  (M_pos={config.mass_pos}, N_neg={config.n_negative}, '
		f'R_bnd={config.boundary_radius})',
		color='white', fontsize=13)

	for ax in axes:
		_style_ax(ax)

	# Panel 1: Density (log scale)
	ax = axes[0]
	ax.fill_between(r_centres, density - density_std, density + density_std,
					alpha=0.3, color='#ff4444')
	ax.plot(r_centres, density, color='#ff4444', lw=1.5, label='ρ(r)')
	ax.axhline(rho_bg, color='#888888', ls='--', lw=1, label=f'ρ_bg = {rho_bg:.4f}')
	ax.set_xlabel('radius')
	ax.set_ylabel('surface density')
	ax.set_title('Density Profile')
	ax.set_yscale('log')
	y_min = max(density[density > 0].min() * 0.3, rho_bg * 1e-4) if np.any(density > 0) else rho_bg * 1e-4
	ax.set_ylim(bottom=y_min, top=rho_bg * 3)
	ax.legend(facecolor='#222222', labelcolor='white', framealpha=0.7, edgecolor='#555555')

	# Panel 2: Normalized density
	ax = axes[1]
	normed = density / rho_bg
	normed_std = density_std / rho_bg
	ax.fill_between(r_centres, normed - normed_std, normed + normed_std,
					alpha=0.3, color='#44aaff')
	ax.plot(r_centres, normed, color='#44aaff', lw=1.5)
	ax.axhline(1.0, color='#888888', ls='--', lw=1, label='ρ/ρ_bg = 1')
	ax.set_xlabel('radius')
	ax.set_ylabel('ρ(r) / ρ_bg')
	ax.set_title('Normalized Density')
	ax.set_ylim(-0.1, 2.0)
	ax.legend(facecolor='#222222', labelcolor='white', framealpha=0.7, edgecolor='#555555')

	# Panel 3: Effective mass
	ax = axes[2]
	ax.fill_between(r_mass, M_eff - M_eff_std, M_eff + M_eff_std,
					alpha=0.3, color='#44ff88')
	ax.plot(r_mass, M_eff, color='#44ff88', lw=1.5, label='M_eff(r)')
	ax.axhline(config.mass_pos * config.n_positive, color='#4488ff', ls='--', lw=1.5,
			   label=f'M_pos = {config.mass_pos * config.n_positive:.0f}')
	ax.axhline(0, color='#555555', ls='-', lw=0.5)
	ax.set_xlabel('radius')
	ax.set_ylabel('effective mass (deficit)')
	ax.set_title('Enclosed Mass Deficit')
	ax.legend(facecolor='#222222', labelcolor='white', framealpha=0.7, edgecolor='#555555')

	plt.tight_layout()
	plt.show()
	return fig


def plot_profile_comparison(histories, labels=None, n_bins=80, equil_fraction=0.5):
	"""
	Compare density profiles from multiple runs (e.g., different N_neg values).

	histories : list of history dicts
	labels    : list of labels for each run
	"""
	if labels is None:
		labels = [f"N_neg={h['config'].n_negative}" for h in histories]

	fig, axes = plt.subplots(1, 3, figsize=(18, 5))
	fig.patch.set_facecolor('#111111')
	fig.suptitle('Cavity Profile Comparison', color='white', fontsize=13)
	for ax in axes:
		_style_ax(ax)

	colors = ['#ff4444', '#ff8844', '#ffff44', '#44ff88', '#44aaff', '#8844ff', '#ff44aa']

	for i, (history, label) in enumerate(zip(histories, labels)):
		config = history['config']
		color = colors[i % len(colors)]

		r_centres, density, _, rho_bg = measure_density_profile(
			history, n_bins=n_bins, equil_fraction=equil_fraction)
		r_mass, M_eff, _ = measure_effective_mass(
			history, n_bins=n_bins, equil_fraction=equil_fraction)

		# Panel 1: Density (log)
		axes[0].plot(r_centres, density, color=color, lw=1.3, label=label)

		# Panel 2: Normalized
		normed = density / rho_bg
		axes[1].plot(r_centres, normed, color=color, lw=1.3, label=label)

		# Panel 3: Effective mass
		axes[2].plot(r_mass, M_eff, color=color, lw=1.3, label=label)

	# Reference lines
	axes[1].axhline(1.0, color='#888888', ls='--', lw=1)
	axes[2].axhline(0, color='#555555', ls='-', lw=0.5)

	# M_pos line (use first history's config)
	m_pos = histories[0]['config'].mass_pos * histories[0]['config'].n_positive
	axes[2].axhline(m_pos, color='#4488ff', ls='--', lw=1.5, label=f'M_pos = {m_pos:.0f}')

	axes[0].set_xlabel('radius')
	axes[0].set_ylabel('surface density')
	axes[0].set_title('Density Profile (log)')
	axes[0].set_yscale('log')

	axes[1].set_xlabel('radius')
	axes[1].set_ylabel('ρ(r) / ρ_bg')
	axes[1].set_title('Normalized Density')

	axes[2].set_xlabel('radius')
	axes[2].set_ylabel('effective mass (deficit)')
	axes[2].set_title('Enclosed Mass Deficit')

	for ax in axes:
		ax.legend(facecolor='#222222', labelcolor='white', framealpha=0.7,
				  edgecolor='#555555', fontsize=9)

	plt.tight_layout()
	plt.show()
	return fig
