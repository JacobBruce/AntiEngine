"""
Phase 3: Cosmological simulation visualization.

Layout (2×2 panels):
  Top-left     : Projected density slice (positive mass — blue)
  Top-right    : Projected density slice (negative mass — red)
  Bottom-left  : Scale factor a(t) and cross-correlation vs time
  Bottom-right : Power spectrum P(k) evolution for both species

The density panels show thin-slab projections through the box center,
revealing the cosmic web structure (filaments + voids).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, LinearSegmentedColormap


# ── Custom colormaps for dark backgrounds ────────────────────────────────────
# Matplotlib's 'Blues'/'Reds' start at near-white, which looks wrong on a dark
# background (voids glow white). These go from near-black through saturated
# color to bright highlights, giving a smooth gradient with many visible levels.
_CMAP_POS = LinearSegmentedColormap.from_list('cosmic_blue', [
	"#0d1346", "#193270", "#2758a1", "#398bc2", "#60aedb", "#82c9df", "#aaddec",
], N=256)
_CMAP_NEG = LinearSegmentedColormap.from_list('cosmic_red', [
	"#381008", "#6D2013", "#a03e25", "#bb6a35", "#cf8252", "#d4a87b", "#ebcaa9",
], N=256)


def _dark_axes(ax):
	ax.set_facecolor('#0d0d0d')
	ax.tick_params(colors='#aaaaaa', labelsize=8)
	ax.xaxis.label.set_color('#cccccc')
	ax.yaxis.label.set_color('#cccccc')
	ax.title.set_color('#ffffff')
	for spine in ax.spines.values():
		spine.set_edgecolor('#333333')


def _project_density(positions, masses, box_size, n_bins=128, slab_frac=0.15):
	"""
	Project particle density through a thin slab centered on the box midplane.

	Sums particle masses in 2D bins (x-y projection) for particles within
	z ∈ [0.5 - slab_frac/2, 0.5 + slab_frac/2] × box_size.

	Returns (n_bins, n_bins) density image.
	"""
	z_center = box_size / 2
	z_half = box_size * slab_frac / 2
	slab_mask = np.abs(positions[:, 2] - z_center) < z_half

	pos_slab = positions[slab_mask]
	m_slab = np.abs(masses[slab_mask])

	density, _, _ = np.histogram2d(
		pos_slab[:, 0], pos_slab[:, 1],
		bins=n_bins, range=[[0, box_size], [0, box_size]],
		weights=m_slab,
	)

	# Normalise by cell area and slab thickness
	cell_area = (box_size / n_bins) ** 2
	slab_thickness = 2 * z_half
	density /= (cell_area * slab_thickness)

	return density


def animate_cosmological(history, interval=100, density_bins=128, slab_frac=0.15):
	"""
	Animate the cosmological simulation.

	history : dict returned by run_cosmological_simulation()
	interval: ms between animation frames
	density_bins: resolution of the density projection
	slab_frac: fraction of box used for the density slab

	Returns (animation, fig) for display.
	"""
	positions_list = history['positions']
	labels = history['labels']
	masses = history['masses']
	box_size = history['box_size']
	times = history['time']
	scale_factors = history['scale_factor']
	hubble_rates = history['hubble_rate']
	cross_corr = history['cross_correlation']
	n_frames = len(positions_list)

	pos_mask = labels == 0
	neg_mask = labels == 1

	# ── Precompute all density projections ───────────────────────────────────
	# Doing this upfront avoids heavy histogram work inside the animation loop,
	# which causes high CPU usage and rendering glitches.
	print(f"Precomputing {n_frames} density frames...", end=' ', flush=True)
	dens_pos_frames = []
	dens_neg_frames = []
	for i in range(n_frames):
		pos = positions_list[i]
		dens_pos_frames.append(
			_project_density(pos[pos_mask], masses[pos_mask], box_size, density_bins, slab_frac)
		)
		dens_neg_frames.append(
			_project_density(pos[neg_mask], masses[neg_mask], box_size, density_bins, slab_frac)
		)
	print("done.")

	# ── Figure setup ─────────────────────────────────────────────────────────
	fig = plt.figure(figsize=(16, 12), facecolor='#111111')
	gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25,
				  left=0.06, right=0.94, top=0.93, bottom=0.06)

	ax_pos_dens = fig.add_subplot(gs[0, 0])
	ax_neg_dens = fig.add_subplot(gs[0, 1])
	ax_scale = fig.add_subplot(gs[1, 0])
	ax_power = fig.add_subplot(gs[1, 1])

	for ax in [ax_pos_dens, ax_neg_dens, ax_scale, ax_power]:
		_dark_axes(ax)

	fig.suptitle('Cosmological Simulation — Phase 3', color='white', fontsize=14)

	# ── Density images ───────────────────────────────────────────────────────
	# Use precomputed first frame; compute global color range across all frames
	dens_pos_0 = dens_pos_frames[0]
	dens_neg_0 = dens_neg_frames[0]

	# Global vmin/vmax across all frames for stable color mapping
	all_pos_nonzero = np.concatenate([d[d > 0].ravel() for d in dens_pos_frames if np.any(d > 0)])
	all_neg_nonzero = np.concatenate([d[d > 0].ravel() for d in dens_neg_frames if np.any(d > 0)])
	vmin_pos = all_pos_nonzero.min() if len(all_pos_nonzero) > 0 else 1e-3
	vmax_pos = all_pos_nonzero.max() if len(all_pos_nonzero) > 0 else 1.0
	vmin_neg = all_neg_nonzero.min() if len(all_neg_nonzero) > 0 else 1e-3
	vmax_neg = all_neg_nonzero.max() if len(all_neg_nonzero) > 0 else 1.0
	box_mpc = box_size / 1000.0
	extent = [0, box_mpc, 0, box_mpc]

	im_pos = ax_pos_dens.imshow(
		dens_pos_0.T, origin='lower', extent=extent, aspect='equal',
		cmap=_CMAP_POS, norm=LogNorm(vmin=vmin_pos, vmax=vmax_pos),
	)
	ax_pos_dens.set_xlabel('x (Mpc)')
	ax_pos_dens.set_ylabel('y (Mpc)')
	title_pos = ax_pos_dens.set_title('Positive mass density')

	im_neg = ax_neg_dens.imshow(
		dens_neg_0.T, origin='lower', extent=extent, aspect='equal',
		cmap=_CMAP_NEG, norm=LogNorm(vmin=vmin_neg, vmax=vmax_neg),
	)
	ax_neg_dens.set_xlabel('x (Mpc)')
	ax_neg_dens.set_ylabel('y (Mpc)')
	title_neg = ax_neg_dens.set_title('Negative mass density')

	# ── Scale factor + cross-correlation panel ───────────────────────────────
	ax_scale.set_xlabel('Time (Gyr)')
	ax_scale.set_ylabel('Scale factor a(t)', color='#66ccff')
	line_a, = ax_scale.plot([], [], color='#66ccff', linewidth=2, label='a(t)')

	# Milne reference: a = a₀ + H₀t (linear expansion for 50/50)
	config = history['config']
	t_ref = np.linspace(0, times[-1], 200)
	a_milne = config.a_initial + config.H0 * t_ref
	ax_scale.plot(t_ref, a_milne, '--', color='#66ccff', alpha=0.4, label='Milne (linear)')

	# Effective scale factor from backreaction measurement
	a_eff = history.get('effective_scale_factor', np.array([]))
	has_a_eff = len(a_eff) > 0 and not np.allclose(a_eff, scale_factors)
	if has_a_eff:
		line_a_eff, = ax_scale.plot([], [], '--', color='#44ff88', linewidth=1.5,
									alpha=0.9, label='a_eff (backreaction)')
	else:
		line_a_eff = None

	ax_scale.set_title('Scale factor & separation')
	ax_scale.legend(loc='upper left', fontsize=8, labelcolor='#cccccc',
					facecolor='#1a1a1a', edgecolor='#333333')

	# Cross-correlation on twin axis
	ax_corr = ax_scale.twinx()
	ax_corr.tick_params(colors='#ff8866', labelsize=8)
	ax_corr.set_ylabel('Cross-correlation', color='#ff8866')
	line_corr, = ax_corr.plot([], [], color='#ff8866', linewidth=1.5, alpha=0.8)
	ax_corr.axhline(0, color='#555555', linewidth=0.5, linestyle=':')
	ax_corr.set_ylim(-1.1, 1.1)

	# ── Power spectrum panel ─────────────────────────────────────────────────
	ax_power.set_xlabel('k (kpc⁻¹)')
	ax_power.set_ylabel('P(k)')
	ax_power.set_xscale('log')
	ax_power.set_yscale('log')
	ax_power.set_title('Power spectrum')

	line_pk_pos, = ax_power.plot([], [], color='#4499ff', linewidth=1.5,
								label='Pos mass', alpha=0.8)
	line_pk_neg, = ax_power.plot([], [], color='#ff4444', linewidth=1.5,
								label='Neg mass', alpha=0.8)
	ax_power.legend(loc='upper right', fontsize=8, labelcolor='#cccccc',
					facecolor='#1a1a1a', edgecolor='#333333')

	# ── Animation function ───────────────────────────────────────────────────

	def update(frame):
		# Use precomputed density projections (no heavy work per frame)
		dens_pos = dens_pos_frames[frame]
		dens_neg = dens_neg_frames[frame]

		im_pos.set_data(dens_pos.T)
		im_neg.set_data(dens_neg.T)

		a_val = scale_factors[frame]
		t_val = times[frame]
		title_pos.set_text(f'Positive mass — a={a_val:.3f}, t={t_val:.2f} Gyr')
		title_neg.set_text(f'Negative mass — a={a_val:.3f}, t={t_val:.2f} Gyr')

		# Scale factor time series
		line_a.set_data(times[:frame + 1], scale_factors[:frame + 1])
		ax_scale.set_xlim(0, max(times[-1], 1))
		a_max = max(scale_factors[-1], a_eff[-1] if has_a_eff else 0) * 1.1
		ax_scale.set_ylim(0, max(a_max, config.a_initial * 2))

		# Effective scale factor (backreaction)
		if line_a_eff is not None and has_a_eff:
			line_a_eff.set_data(times[:frame + 1], a_eff[:frame + 1])

		# Cross-correlation
		line_corr.set_data(times[:frame + 1], cross_corr[:frame + 1])

		# Power spectra (use nearest recorded spectrum)
		pk_pos_list = history['pk_pos']
		pk_neg_list = history['pk_neg']
		k_bins = history['k_bins']
		if len(pk_pos_list) > 0 and k_bins is not None:
			# Map frame index to nearest power spectrum recording
			pk_idx = min(frame // 5, len(pk_pos_list) - 1)
			pk_p = pk_pos_list[pk_idx]
			pk_n = pk_neg_list[pk_idx]

			valid_p = pk_p > 0
			valid_n = pk_n > 0
			if np.any(valid_p):
				line_pk_pos.set_data(k_bins[valid_p], pk_p[valid_p])
			if np.any(valid_n):
				line_pk_neg.set_data(k_bins[valid_n], pk_n[valid_n])

			all_pk = np.concatenate([pk_p[valid_p], pk_n[valid_n]]) if (np.any(valid_p) or np.any(valid_n)) else np.array([1.0])
			if len(all_pk) > 0:
				ax_power.set_xlim(k_bins[1], k_bins[-1])
				ax_power.set_ylim(max(all_pk.min() * 0.1, 1e-10), all_pk.max() * 10)

		return im_pos, im_neg, line_a, line_a_eff, line_corr, line_pk_pos, line_pk_neg

	ani = animation.FuncAnimation(
		fig, update, frames=n_frames, interval=interval, blit=False,
		repeat=False,
	)

	# ── Playback controls ────────────────────────────────────────────────────
	# Space: play/pause, R: restart, Left/Right: step when paused
	playback = {'paused': False, 'frame': 0}

	def _ensure_event_source():
		"""Re-create the timer if the animation has finished (repeat=False sets it to None)."""
		if ani.event_source is None:
			ani.event_source = fig.canvas.new_timer(interval=interval)
			ani.event_source.add_callback(ani._step)

	def on_key(event):
		if event.key == ' ':
			_ensure_event_source()
			if playback['paused']:
				ani.event_source.start()
				playback['paused'] = False
			else:
				ani.event_source.stop()
				playback['paused'] = True
		elif event.key == 'r':
			# Restart from frame 0
			ani.frame_seq = ani.new_frame_seq()
			_ensure_event_source()
			ani.event_source.start()
			playback['paused'] = False
			update(0)
			fig.canvas.draw_idle()
		elif event.key == 'right' and playback['paused']:
			playback['frame'] = min(playback['frame'] + 1, n_frames - 1)
			update(playback['frame'])
			fig.canvas.draw_idle()
		elif event.key == 'left' and playback['paused']:
			playback['frame'] = max(playback['frame'] - 1, 0)
			update(playback['frame'])
			fig.canvas.draw_idle()

	fig.canvas.mpl_connect('key_press_event', on_key)
	print("Playback controls: [Space] play/pause, [R] restart, [←/→] step when paused")

	plt.show()
	return ani, fig


def plot_density_snapshot(history, frame=-1, density_bins=128, slab_frac=0.15):
	"""
	Plot a static high-resolution density comparison (pos vs neg mass).
	"""
	if frame < 0:
		frame = len(history['positions']) + frame

	positions = history['positions'][frame]
	labels = history['labels']
	masses = history['masses']
	box_size = history['box_size']
	a_val = history['scale_factor'][frame]
	t_val = history['time'][frame]

	pos_mask = labels == 0
	neg_mask = labels == 1

	dens_pos = _project_density(positions[pos_mask], masses[pos_mask],
								box_size, density_bins, slab_frac)
	dens_neg = _project_density(positions[neg_mask], masses[neg_mask],
								box_size, density_bins, slab_frac)

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#111111')
	for ax in [ax1, ax2]:
		_dark_axes(ax)

	box_mpc = box_size / 1000.0
	extent = [0, box_mpc, 0, box_mpc]

	vmin_p = max(dens_pos[dens_pos > 0].min(), 1e-6) if np.any(dens_pos > 0) else 1e-6
	vmin_n = max(dens_neg[dens_neg > 0].min(), 1e-6) if np.any(dens_neg > 0) else 1e-6

	ax1.imshow(dens_pos.T, origin='lower', extent=extent, aspect='equal',
			   cmap=_CMAP_POS, norm=LogNorm(vmin=vmin_p, vmax=dens_pos.max()))
	ax1.set_title(f'Positive mass — a={a_val:.3f}, t={t_val:.2f} Gyr', color='white')
	ax1.set_xlabel('x (Mpc)')
	ax1.set_ylabel('y (Mpc)')

	ax2.imshow(dens_neg.T, origin='lower', extent=extent, aspect='equal',
			   cmap=_CMAP_NEG, norm=LogNorm(vmin=vmin_n, vmax=dens_neg.max()))
	ax2.set_title(f'Negative mass — a={a_val:.3f}, t={t_val:.2f} Gyr', color='white')
	ax2.set_xlabel('x (Mpc)')
	ax2.set_ylabel('y (Mpc)')

	fig.suptitle('Cosmological Density Projection', color='white', fontsize=14)
	plt.tight_layout()
	plt.show()
	return fig
