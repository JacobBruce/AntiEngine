"""
Phase 2B: Galaxy cluster visualization.

Layout (2×2 panels):
  Top-left     : x-y cluster scatter (animated) — particles colored by galaxy
  Top-right    : All pairwise separations d_ij(t) — full time series + moving marker
  Bottom-left  : Mean separation ⟨d⟩(t) — full time series + moving marker
  Bottom-right : Energy conservation ΔE/E₀(t) — full time series + moving marker

The scatter panel is fully animated (particles move).
The three plot panels show static time-series curves with an animated vertical
marker tracking the current simulation time.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from scipy.integrate import odeint


# ─────────────────────────────────────────────────────────────────────────────
# CPL Hubble reference
# ─────────────────────────────────────────────────────────────────────────────

def _cpl_hubble_reference(
	d0: float,
	times: np.ndarray,
	H0: float,
	w0: float = -1.0,
	wa: float = 0.0,
) -> np.ndarray:
	"""
	Compute the Hubble-flow reference separation d(t) under the CPL dark
	energy equation-of-state w(a) = w0 + wa*(1-a).

	Uses a dark-energy-dominated (de Sitter base) Friedmann:
	  H²(a)/H₀² = a^(-3(1+w0+wa)) × exp(-3·wa·(1-a))

	This is equivalent to Ωm=0, Ω_DE=1 — consistent with the existing
	d₀·exp(H₀·t) reference.  ΛCDM (w0=-1, wa=0) recovers d₀·exp(H₀·t) exactly.

	CPL DESI results (dark energy weaker today than in the past) cause the
	reference to fall below ΛCDM at late times (>~5 Gyr).
	"""
	def dadt(a, t):
		if a <= 0:
			return 0.0
		# H²/H₀² = a^(-3(1+w0+wa)) * exp(-3*wa*(1-a))
		exponent = -3.0 * (1.0 + w0 + wa)
		H_rel    = a ** (exponent / 2.0) * np.exp(-1.5 * wa * (1.0 - a))
		return a * H0 * H_rel

	t_max   = float(times[-1]) if len(times) > 0 else 1.0
	t_dense = np.linspace(0.0, t_max, max(2000, len(times) * 10))
	a_dense = odeint(dadt, 1.0, t_dense, rtol=1e-10, atol=1e-12).ravel()

	a_interp = np.interp(times, t_dense, a_dense)
	return d0 * a_interp / a_interp[0]


# Galaxy colors (up to 8 galaxies)
_GAL_COLOURS = [
	'#ff4444',  # red
	'#ffaa44',  # orange
	'#44ff88',  # green
	'#44aaff',  # blue
	'#ff44ff',  # magenta
	'#ffff44',  # yellow
	'#44ffff',  # cyan
	'#ff8844',  # amber
]
_NEG_COLOUR = '#2255aa'  # dark blue for neg-mass particles


def _dark_axes(ax):
	ax.set_facecolor('#1a1a1a')
	ax.tick_params(colors='#aaaaaa', labelsize=7)
	for spine in ax.spines.values():
		spine.set_edgecolor('#444444')
	ax.xaxis.label.set_color('#cccccc')
	ax.yaxis.label.set_color('#cccccc')
	ax.title.set_color('#cccccc')


def animate_cluster(history, interval=60, view_range=None):
	"""
	Produce an animated visualization of the galaxy cluster simulation.

	Parameters
	----------
	history     : dict returned by run_cluster_simulation
	interval    : animation frame interval in ms
	view_range  : half-width of the x-y scatter view in kpc.
	              Defaults to cluster_radius × 1.1.
	"""
	config   = history['config']
	times    = np.array(history['time'])
	n_frames = len(times)
	n_gal    = config.n_galaxies
	labels   = history['labels']      # (N_total,) int32
	gal_ids  = history['galaxy_ids']  # (N_total,) int32

	if view_range is None:
		view_range = config.cluster_radius * 1.1

	# Pre-compute per-pair separation arrays for static line plots
	n_pairs   = n_gal * (n_gal - 1) // 2
	pair_seps = np.array(history['pairwise_separations'])   # (n_frames, n_pairs)
	mean_seps = np.array(history['mean_separation'])        # (n_frames,)
	E         = np.array(history['E'])
	E0        = E[0]
	rel_E     = (E - E0) / abs(E0) * 100

	# ── Figure layout ─────────────────────────────────────────────────────
	fig = plt.figure(figsize=(13, 8), facecolor='#111111')
	gs  = fig.add_gridspec(2, 2, left=0.06, right=0.96, top=0.95, bottom=0.10,
	                       hspace=0.30, wspace=0.25)
	ax_xy   = fig.add_subplot(gs[0, 0])
	ax_pair = fig.add_subplot(gs[0, 1])
	ax_mean = fig.add_subplot(gs[1, 0])
	ax_en   = fig.add_subplot(gs[1, 1])

	for ax in [ax_xy, ax_pair, ax_mean, ax_en]:
		_dark_axes(ax)

	# ── Static: pairwise separations ──────────────────────────────────────
	pair_cmap    = plt.cm.tab20 if n_pairs > 10 else plt.cm.Set1
	pair_colours = [pair_cmap(p / max(n_pairs - 1, 1)) for p in range(n_pairs)]
	show_pair_legend = n_pairs <= 16
	k = 0
	for i in range(n_gal):
		for j in range(i + 1, n_gal):
			lbl = f'{i}–{j}' if show_pair_legend else None
			ax_pair.plot(
				times, pair_seps[:, k],
				color=pair_colours[k], lw=1.2, alpha=0.8, label=lbl,
			)
			k += 1
	ax_pair.set_xlabel('Time [Gyr]')
	ax_pair.set_ylabel('Separation [kpc]')
	ax_pair.set_title('Galaxy pairwise separations')
	if show_pair_legend:
		ax_pair.legend(
			facecolor='#222222', labelcolor='white', fontsize=6,
			framealpha=0.7, edgecolor='#444444',
			ncols=max(1, n_pairs // 8),
		)
	pair_marker = ax_pair.axvline(times[0], color='#ffffff', lw=1.0, alpha=0.7)

	# ── Static: mean separation ───────────────────────────────────────────
	# Hubble H₀ = 70 km/s/Mpc in simulation units:
	# 70 × 1.0227 kpc/Gyr per km/s / 1000 kpc per Mpc = 0.07159 Gyr⁻¹
	#
	# Reference curve: integrate the CPL Friedmann equation with config.w0/wa.
	# Uses Ωm=0.3, Ω_DE=0.7 flat ΛCDM background with CPL dark energy EOS.
	# ΛCDM (w0=-1, wa=0) recovers the pure de Sitter exponential exactly.
	H0_per_Gyr = 70.0 * 1.0227 / 1000.0   # ≈ 0.07159 Gyr⁻¹
	d0         = mean_seps[0]
	d_hubble   = _cpl_hubble_reference(d0, times, H0_per_Gyr, config.w0, config.wa)
	# Build legend label: show ΛCDM if w0/wa unset, else show CPL params
	w0, wa = config.w0, config.wa
	if w0 == -1.0 and wa == 0.0:
		hub_label = 'Hubble H₀=70 km/s/Mpc (ΛCDM)'
	else:
		hub_label = f'Hubble CPL w₀={w0}, wₐ={wa}'

	ax_mean.plot(times, mean_seps, color='#88ff88', lw=1.5, label='Simulation')
	ax_mean.plot(times, d_hubble,  color='#ffff44', lw=1.2, ls='--', label=hub_label)
	ax_mean.set_xlabel('Time [Gyr]')
	ax_mean.set_ylabel('⟨d⟩ [kpc]')
	ax_mean.set_title('Mean galaxy separation')
	ax_mean.legend(
		facecolor='#222222', labelcolor='white', fontsize=6,
		framealpha=0.7, edgecolor='#444444', loc='upper left',
	)
	mean_marker = ax_mean.axvline(times[0], color='#ffffff', lw=1.0, alpha=0.7)

	# ── Static: energy conservation ──────────────────────────────────────
	ax_en.plot(times, rel_E, color='#ff8844', lw=1.2)
	ax_en.axhline(0, color='#444444', lw=0.8)
	ax_en.set_xlabel('Time [Gyr]')
	ax_en.set_ylabel('ΔE / E₀ [%]')
	ax_en.set_title('Energy conservation')
	en_marker = ax_en.axvline(times[0], color='#ffffff', lw=1.0, alpha=0.7)

	# ── Scatter: cluster x-y (animated) ──────────────────────────────────
	ax_xy.set_xlim(-view_range, view_range)
	ax_xy.set_ylim(-view_range, view_range)
	ax_xy.set_aspect('equal')
	ax_xy.set_xlabel('x [kpc]')
	ax_xy.set_ylabel('y [kpc]')
	ax_xy.set_title('Galaxy cluster (x-y projection)')

	neg_mask = labels == 4
	scat_neg  = ax_xy.scatter(
		[], [], c=_NEG_COLOUR, s=0.8, alpha=0.25,
		rasterized=True, zorder=1, label='neg mass',
	)
	scat_stars = []  # disk + bulge per galaxy
	scat_bh    = []  # BH per galaxy
	for g in range(n_gal):
		col = _GAL_COLOURS[g % len(_GAL_COLOURS)]
		# Only label first 8 galaxies in the scatter legend to avoid overflow
		gal_label = f'Galaxy {g}' if g < 7 else ('' if g > 7 else '_nolegend_')
		sc = ax_xy.scatter(
			[], [], c=col, s=2.5, alpha=0.85,
			rasterized=True, zorder=3, label=gal_label,
		)
		bh = ax_xy.scatter(
			[], [], c=col, s=50, marker='*',
			zorder=5,
		)
		scat_stars.append(sc)
		scat_bh.append(bh)

	ax_xy.legend(
		facecolor='#222222', labelcolor='white', fontsize=6,
		framealpha=0.7, edgecolor='#444444',
		loc='upper right', ncols=2 if n_gal > 4 else 1,
	)
	time_text = ax_xy.text(
		0.02, 0.97, '', transform=ax_xy.transAxes,
		color='white', fontsize=8, va='top',
		bbox=dict(facecolor='#111111', alpha=0.7, edgecolor='none'),
	)

	# ── Zoom slider ───────────────────────────────────────────────────────
	max_range = view_range * 1.5
	boundary_radii_arr = np.array(history.get('boundary_radius', []))
	if len(boundary_radii_arr) > 0:
		max_range = max(max_range, float(boundary_radii_arr.max()) * 1.2)
	ax_slider = fig.add_axes([0.06, 0.02, 0.30, 0.02], facecolor='#222222')
	zoom_slider = Slider(
		ax_slider, 'Zoom', 50.0, max_range,
		valinit=view_range,
		valstep=10.0,
		color='#4499ff',
	)
	ax_slider.xaxis.label.set_color('#aaaaaa')
	zoom_slider.label.set_color('#aaaaaa')
	zoom_slider.valtext.set_color('#cccccc')

	def on_zoom(val):
		ax_xy.set_xlim(-val, val)
		ax_xy.set_ylim(-val, val)
		fig.canvas.draw_idle()

	zoom_slider.on_changed(on_zoom)

	# ── Animation update function ─────────────────────────────────────────
	def update(frame_idx):
		pos = history['positions'][frame_idx]
		t   = history['time'][frame_idx]

		# Neg mass
		scat_neg.set_offsets(pos[neg_mask, :2])

		# Per-galaxy stellar particles and BH
		for g in range(n_gal):
			star_mask = (gal_ids == g) & (labels != 5)
			bh_mask   = (gal_ids == g) & (labels == 5)
			scat_stars[g].set_offsets(pos[star_mask, :2])
			if bh_mask.any():
				scat_bh[g].set_offsets(pos[bh_mask, :2])
			else:
				scat_bh[g].set_offsets(np.empty((0, 2)))

		# Advance time markers
		for mk in [pair_marker, mean_marker, en_marker]:
			mk.set_xdata([t, t])

		time_text.set_text(f't = {t:.3f} Gyr')

		return [
			scat_neg, *scat_stars, *scat_bh,
			pair_marker, mean_marker, en_marker,
			time_text,
		]

	ani = animation.FuncAnimation(
		fig, update,
		frames=n_frames,
		interval=interval,
		blit=True,
	)
	plt.show()
	return ani, zoom_slider  # keep slider reference to prevent GC
