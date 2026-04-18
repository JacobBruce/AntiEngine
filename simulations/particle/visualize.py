"""
Visualisation for the particle-scale simulation.

Provides a real-time animated scatter plot (side-by-side with an energy panel)
using pre-recorded simulation history. Positive-mass particles are shown in blue,
negative-mass particles in red.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_simulation(history, interval: int = 50, steps_per_frame: int = 1):
	"""
	Show an animated scatter plot of the recorded simulation.

	history         : dict returned by run_simulation()
	interval        : milliseconds between animation frames
	steps_per_frame : subsample recorded frames (1 = show every frame)

	Returns the FuncAnimation object (keep a reference to prevent GC).
	"""
	positions_list = history['positions'][::steps_per_frame]
	masses         = history['masses']
	times          = np.array(history['time'][::steps_per_frame])
	KE_vals        = np.array(history['KE'][::steps_per_frame])
	PE_vals        = np.array(history['PE'][::steps_per_frame])
	E_vals         = np.array(history['total_E'][::steps_per_frame])
	E0             = history['total_E'][0]

	pos_mask = masses > 0
	neg_mask = masses < 0

	# Determine axis limits from the full trajectory
	all_pos = np.array(positions_list)
	lim = np.max(np.abs(all_pos)) * 1.1 + 1.0

	# ── Layout ──────────────────────────────────────────────────────────────
	fig, (ax_sim, ax_energy) = plt.subplots(1, 2, figsize=(12, 5))
	fig.patch.set_facecolor('#111111')

	for ax in (ax_sim, ax_energy):
		ax.set_facecolor('#111111')
		ax.tick_params(colors='#cccccc')
		ax.xaxis.label.set_color('#cccccc')
		ax.yaxis.label.set_color('#cccccc')
		ax.title.set_color('#ffffff')
		for spine in ax.spines.values():
			spine.set_edgecolor('#444444')

	# ── Simulation panel ────────────────────────────────────────────────────
	ax_sim.set_xlim(-lim, lim)
	ax_sim.set_ylim(-lim, lim)
	ax_sim.set_aspect('equal')
	ax_sim.set_xlabel('x')
	ax_sim.set_ylabel('y')
	ax_sim.set_title('Particle Simulation')

	scat_pos = ax_sim.scatter([], [], c='#4488ff', s=18, alpha=0.85, label='+mass')
	scat_neg = ax_sim.scatter([], [], c='#ff4444', s=18, alpha=0.85, label='−mass')
	time_text = ax_sim.text(
		0.02, 0.97, '', transform=ax_sim.transAxes,
		color='white', va='top', fontsize=9, family='monospace',
	)
	ax_sim.legend(
		loc='upper right', facecolor='#222222',
		labelcolor='white', framealpha=0.7, edgecolor='#555555',
	)

	# ── Energy panel ────────────────────────────────────────────────────────
	ax_energy.plot(times, KE_vals,        color='#4488ff', lw=1.2, alpha=0.8, label='KE')
	ax_energy.plot(times, PE_vals,        color='#ff4444', lw=1.2, alpha=0.8, label='PE')
	ax_energy.plot(times, E_vals - E0,    color='#ffffff', lw=1.5,             label='ΔE (total)')
	ax_energy.axhline(0, color='#555555', lw=0.8, ls='--')
	ax_energy.set_xlabel('time')
	ax_energy.set_ylabel('energy')
	ax_energy.set_title('Energy')
	ax_energy.legend(
		facecolor='#222222', labelcolor='white',
		framealpha=0.7, edgecolor='#555555',
	)

	# Vertical marker that tracks the current animation frame
	energy_marker = ax_energy.axvline(times[0], color='#ffff44', lw=1.2, alpha=0.8)

	# ── Animation update ────────────────────────────────────────────────────
	def update(frame_idx):
		pos = positions_list[frame_idx]
		scat_pos.set_offsets(pos[pos_mask])
		scat_neg.set_offsets(pos[neg_mask])
		time_text.set_text(f't = {times[frame_idx]:.2f}')
		energy_marker.set_xdata([times[frame_idx]])
		return scat_pos, scat_neg, time_text, energy_marker

	ani = animation.FuncAnimation(
		fig, update,
		frames=len(positions_list),
		interval=interval,
		blit=True,
	)

	plt.tight_layout()
	plt.show()
	return ani


def plot_snapshot(positions, masses, title: str = 'Snapshot', ax=None):
	"""Plot a single frame — useful for quick inspection."""
	if ax is None:
		_, ax = plt.subplots(figsize=(6, 6))

	pos_mask = masses > 0
	neg_mask = masses < 0

	ax.scatter(positions[pos_mask, 0], positions[pos_mask, 1],
	           c='#4488ff', s=20, alpha=0.85, label='+mass')
	ax.scatter(positions[neg_mask, 0], positions[neg_mask, 1],
	           c='#ff4444', s=20, alpha=0.85, label='−mass')
	ax.set_aspect('equal')
	ax.legend()
	ax.set_title(title)
	return ax
