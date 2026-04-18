"""
Visualisation for the single-galaxy simulation.

Layout: 3-panel figure
  Left   : 3D scatter of all particles (x-y projection + y-z inset)
  Middle : Rotation curve: v_phi(R) at multiple simulation times
  Right  : Cavity radius and energy over time
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider

# Particle colour scheme
_COLOURS = {
	0: '#4499ff',   # disk       — blue
	1: '#ffcc44',   # bulge      — gold
	2: '#99ddff',   # st. halo   — light blue
	3: '#55ddaa',   # gas        — teal
	4: '#ff4444',   # neg. mass  — red
	5: '#ffffff',   # black hole — white star
}
_LABELS = {
	0: 'Disk',
	1: 'Bulge',
	2: 'St. halo',
	3: 'Gas halo',
	4: 'Neg. mass',
	5: 'Black hole',
}
_SIZES = {0: 8, 1: 10, 2: 4, 3: 4, 4: 5, 5: 40}


def _dark_axes(ax):
	ax.set_facecolor('#0d0d0d')
	ax.tick_params(colors='#aaaaaa', labelsize=8)
	ax.xaxis.label.set_color('#cccccc')
	ax.yaxis.label.set_color('#cccccc')
	ax.title.set_color('#ffffff')
	for spine in ax.spines.values():
		spine.set_edgecolor('#333333')


def animate_galaxy(history, interval: int = 80, view_range: float = None):
	"""
	Animate the galaxy simulation.

	history    : dict returned by run_galaxy_simulation()
	interval   : milliseconds between frames
	view_range : half-range of the x-y scatter plot in kpc; default = auto
	"""
	positions_list = history['positions']
	labels_arr     = history['labels']
	masses_arr     = history['masses']
	times          = np.array(history['time'])
	cavities       = np.array(history['cavity_radius'])

	if view_range is None:
		# Show out to 90th percentile of positive-mass positions
		pos_mask = masses_arr > 0
		all_pos  = np.concatenate([p[pos_mask] for p in positions_list], axis=0)
		view_range = float(np.percentile(np.abs(all_pos[:, :2]), 95)) * 1.2
		view_range = max(view_range, 20.0)

	# ── Figure layout ────────────────────────────────────────────────────────
	fig = plt.figure(figsize=(16, 8.5), facecolor='#111111')
	fig.suptitle('AntiEngine — Single Galaxy Simulation', color='white', fontsize=12)
	# Main panel grid with bottom margin for zoom slider
	gs  = GridSpec(2, 3, figure=fig,
	               left=0.05, right=0.95, top=0.93, bottom=0.10,
	               hspace=0.35, wspace=0.30)

	ax_xy   = fig.add_subplot(gs[:, 0])   # x-y projection (tall)
	ax_yz   = fig.add_subplot(gs[0, 1])   # y-z projection
	ax_rc   = fig.add_subplot(gs[1, 1])   # rotation curve history (combined)
	ax_cav  = fig.add_subplot(gs[0, 2])   # cavity radius vs time
	ax_eff  = fig.add_subplot(gs[1, 2])   # effective positive mass from cavity

	for ax in (ax_xy, ax_yz, ax_rc, ax_cav, ax_eff):
		_dark_axes(ax)

	# ── Static panels ────────────────────────────────────────────────────────
	# Cavity radius (full time series, with optional boundary radius overlay)
	ax_cav.plot(times, cavities, color='#ff8844', lw=1.5, label='Cavity')
	boundary_radii = np.array(history.get('boundary_radius', []))
	if len(boundary_radii) > 0 and boundary_radii[-1] != boundary_radii[0]:
		ax_cav.plot(times, boundary_radii, color='#aaaaaa', lw=1.0, ls='--',
		            label='Boundary', alpha=0.7)
		ax_cav.legend(facecolor='#222222', labelcolor='white', fontsize=6,
		              framealpha=0.7, edgecolor='#444444')
	ax_cav.set_xlabel('Time [Gyr]')
	ax_cav.set_ylabel('Radius [kpc]')
	ax_cav.set_title('Neg. mass cavity')
	cav_marker = ax_cav.axvline(times[0], color='#ffffff', lw=1.0, alpha=0.7)

	# Effective positive mass (full time series)
	eff_mass = np.array(history.get('effective_mass', [np.nan] * len(times)))
	total_pos_mass = history['config'].total_pos_mass
	eff_frac = eff_mass / total_pos_mass  # as fraction of total stellar mass
	ax_eff.plot(times, eff_frac, color='#88ff88', lw=1.5)
	ax_eff.axhline(0, color='#444444', lw=0.8)
	ax_eff.set_xlabel('Time [Gyr]')
	ax_eff.set_ylabel('M_eff / M_stellar')
	ax_eff.set_title('Eff. dark mass (cavity)')
	eff_marker = ax_eff.axvline(times[0], color='#ffffff', lw=1.0, alpha=0.7)

	# Rotation curve history — all frames pre-drawn in plasma colour gradient
	rcs    = history['rotation_curves']
	ercs   = history.get('effective_rotation_curves', [])
	prcs   = history.get('pos_only_rotation_curves', [])
	n_frms = len(rcs)
	cmap_rc = plt.cm.plasma
	for k, rc in enumerate(rcs):
		good = rc['n'] > 0
		col  = cmap_rc(k / max(n_frms - 1, 1))
		ax_rc.plot(rc['r'][good], rc['v'][good], color=col, lw=0.8, alpha=0.45)
	for k, erc in enumerate(ercs):
		col = cmap_rc(k / max(n_frms - 1, 1))
		ax_rc.plot(erc['r'], erc['v'], color=col, lw=0.6, alpha=0.25, ls=':')
	for k, prc in enumerate(prcs):
		col = cmap_rc(k / max(n_frms - 1, 1))
		ax_rc.plot(prc['r'], prc['v'], color=col, lw=0.5, alpha=0.15, ls='--')
	# Animated lines: white=measured particle vφ; cyan dashed=v_c with cavity dark mass; orange dotted=stellar-only v_c
	rc_cur, = ax_rc.plot([], [], color='white',   lw=2.0, zorder=5, label='v_φ (particles)')
	rc_eff, = ax_rc.plot([], [], color='#44ffcc', lw=1.5, zorder=4, ls='--', label='v_c (cavity)')
	rc_pos, = ax_rc.plot([], [], color='#ffaa44', lw=1.5, zorder=3, ls=':',  label='v_c (stars only)')
	ax_rc.axhline(220, color='#888888', lw=1.0, ls=':', alpha=0.6, label='~220 km/s')
	ax_rc.set_xlabel('R [kpc]')
	ax_rc.set_ylabel('v_φ [km/s]')
	ax_rc.set_title('Rotation curve history')
	ax_rc.set_xlim(0, 80)
	ax_rc.set_ylim(0, 300)
	ax_rc.legend(facecolor='#222222', labelcolor='white', fontsize=6,
	             framealpha=0.7, edgecolor='#444444')
	# Colorbar indicating time direction
	sm = plt.cm.ScalarMappable(cmap=cmap_rc,
	                           norm=plt.Normalize(vmin=times[0], vmax=times[-1]))
	sm.set_array([])
	cb = fig.colorbar(sm, ax=ax_rc, pad=0.02, fraction=0.046)
	cb.set_label('Time [Gyr]', color='#cccccc', fontsize=7)
	cb.ax.yaxis.set_tick_params(colors='#aaaaaa', labelsize=6)

	# ── Scatter plot objects (animated) ─────────────────────────────────────
	scatter_objects_xy = {}
	scatter_objects_yz = {}
	for lid, colour in _COLOURS.items():
		s      = _SIZES[lid]
		alpha  = 1.0 if lid == 5 else 0.75
		marker = '*' if lid == 5 else 'o'
		zorder = 10 if lid == 5 else 1
		scatter_objects_xy[lid] = ax_xy.scatter(
			[], [], c=colour, s=s, alpha=alpha, marker=marker,
			zorder=zorder, label=_LABELS[lid], rasterized=True
		)
		scatter_objects_yz[lid] = ax_yz.scatter(
			[], [], c=colour, s=s, alpha=alpha, marker=marker,
			zorder=zorder, rasterized=True
		)

	ax_xy.set_xlim(-view_range, view_range)
	ax_xy.set_ylim(-view_range, view_range)
	ax_xy.set_aspect('equal')
	ax_xy.set_xlabel('x [kpc]')
	ax_xy.set_ylabel('y [kpc]')
	ax_xy.set_title('x–y projection')
	ax_xy.legend(loc='upper right', facecolor='#222222', labelcolor='white',
	             fontsize=7, framealpha=0.7, edgecolor='#444444', markerscale=1.5)
	time_text = ax_xy.text(
		0.02, 0.97, '', transform=ax_xy.transAxes,
		color='white', va='top', fontsize=9, family='monospace',
	)

	# Square y-z: match xlim/ylim range so equal-aspect fills the column width
	# identically to the other panels below it (which have no aspect constraint).
	ax_yz.set_xlim(-view_range, view_range)
	ax_yz.set_ylim(-view_range, view_range)
	ax_yz.set_aspect('equal')
	ax_yz.set_xlabel('y [kpc]')
	ax_yz.set_ylabel('z [kpc]')
	ax_yz.set_title('y–z (edge-on)')

	# ── Zoom slider ──────────────────────────────────────────────────────────
	max_range = history['config'].neg_sphere_radius * 1.2
	boundary_radii_arr = np.array(history.get('boundary_radius', []))
	if len(boundary_radii_arr) > 0:
		max_range = max(max_range, float(boundary_radii_arr.max()) * 1.2)
	ax_slider = fig.add_axes([0.05, 0.02, 0.30, 0.02], facecolor='#222222')
	zoom_slider = Slider(
		ax_slider, 'Zoom', 5.0, max_range,
		valinit=view_range,
		valstep=1.0,
		color='#4499ff',
	)
	ax_slider.xaxis.label.set_color('#aaaaaa')
	zoom_slider.label.set_color('#aaaaaa')
	zoom_slider.valtext.set_color('#cccccc')

	def on_zoom(val):
		ax_xy.set_xlim(-val, val)
		ax_xy.set_ylim(-val, val)
		ax_yz.set_xlim(-val, val)
		ax_yz.set_ylim(-val, val)
		fig.canvas.draw_idle()

	zoom_slider.on_changed(on_zoom)

	# ── Animation update ─────────────────────────────────────────────────────
	def update(frame_idx):
		pos = positions_list[frame_idx]

		# Update scatter positions
		for lid in _COLOURS:
			mask = labels_arr == lid
			scatter_objects_xy[lid].set_offsets(pos[mask, :2])
			scatter_objects_yz[lid].set_offsets(pos[mask, 1:3])

		time_text.set_text(f't = {times[frame_idx]:.3f} Gyr')

		# Update current-frame rotation curves (particle measured + effective analytical + stellar-only)
		rc = history['rotation_curves'][frame_idx]
		good = rc['n'] > 0
		rc_cur.set_data(rc['r'][good], rc['v'][good])
		ercs_list = history.get('effective_rotation_curves', [])
		if ercs_list:
			erc = ercs_list[frame_idx]
			rc_eff.set_data(erc['r'], erc['v'])
		prcs_list = history.get('pos_only_rotation_curves', [])
		if prcs_list:
			prc = prcs_list[frame_idx]
			rc_pos.set_data(prc['r'], prc['v'])

		# Time markers on cavity and effective-mass panels
		t = times[frame_idx]
		cav_marker.set_xdata([t])
		eff_marker.set_xdata([t])

		return (
			*scatter_objects_xy.values(),
			*scatter_objects_yz.values(),
			time_text, rc_cur, rc_eff, rc_pos, cav_marker, eff_marker,
		)

	ani = animation.FuncAnimation(
		fig, update,
		frames=len(positions_list),
		interval=interval,
		blit=True,
	)

	plt.show()
	return ani, zoom_slider  # keep slider reference to prevent GC


def plot_rotation_curve_comparison(history, time_indices=None):
	"""
	Static plot of rotation curves at several simulation times.

	time_indices : list of frame indices to plot; default = evenly spaced 5 frames
	"""
	times = np.array(history['time'])
	rcs   = history['rotation_curves']

	if time_indices is None:
		n = len(rcs)
		time_indices = [int(i * (n - 1) / 4) for i in range(5)]

	fig, ax = plt.subplots(figsize=(8, 5))
	fig.patch.set_facecolor('#111111')
	_dark_axes(ax)

	cmap = plt.cm.plasma
	for k, idx in enumerate(time_indices):
		rc   = rcs[idx]
		good = rc['n'] > 0
		col  = cmap(k / max(len(time_indices) - 1, 1))
		ax.plot(rc['r'][good], rc['v'][good], color=col, lw=1.8,
		        label=f't = {times[idx]:.2f} Gyr')
		ax.fill_between(
			rc['r'][good],
			rc['v'][good] - rc['v_std'][good],
			rc['v'][good] + rc['v_std'][good],
			color=col, alpha=0.15,
		)

	ax.set_xlabel('R [kpc]')
	ax.set_ylabel('v_φ [km/s]')
	ax.set_title('Rotation curves over time')
	ax.axhline(220, color='#888888', lw=1.0, ls=':', alpha=0.6, label='MW observed (~220 km/s)')
	ax.legend(facecolor='#222222', labelcolor='white', fontsize=8,
	          framealpha=0.8, edgecolor='#444444')
	ax.set_xlim(0, 80)
	ax.set_ylim(0, 300)
	plt.tight_layout()
	plt.show()
	return fig, ax
