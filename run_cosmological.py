"""
Entry point for the Phase 3 cosmological simulation.

Tests the anti-universe model at cosmological scale with pre-separated
(anti-correlated) initial conditions:
  - Positive and negative mass start from INVERTED perturbation fields
  - Where pos has an overdensity, neg has an underdensity (and vice versa)
  - This represents the state AFTER the rapid early-universe separation
  - Displacements are scaled by D(a_initial)/D(1) using the anti-universe
    growth factor, making results independent of a_initial
  - Scale factor evolves self-consistently from the Friedmann equation

Key observables:
  - Density projections showing cosmic web vs anti-cosmic web
  - Cross-correlation between pos and neg density fields
    (should start near -1.0 and evolve as structure forms)
  - Power spectrum evolution: P(k) should grow for pos, decay for neg
  - Scale factor a(t): for 50/50, expect Milne-like linear expansion

Run from the project root:
    python run_cosmological.py
"""

from simulations.cosmological.initial_conditions import CosmoConfig
from simulations.cosmological.sim import run_cosmological_simulation
from simulations.cosmological.visualize import animate_cosmological
from simulations.cosmological.visualize import plot_density_snapshot

config = CosmoConfig(
	# ── Box and resolution ───────────────────────────────────────────────────
	box_size = 100_000.0,      # 100 Mpc comoving box
	n_grid = 64,               # PM grid cells per dimension (ideally 2× n_per_dim)
	n_per_dim = 32,            # 32³ = 32768 particles per species

	# ── Mass ratio ───────────────────────────────────────────────────────────
	# 50/50 (zero-energy universe) — matched DESI CPL in cluster sim
	#neg_mass_ratio = 1.0,
	neg_mass_ratio = 7.14 / 2.86, # dark energy ratio

	# ── Cosmology ────────────────────────────────────────────────────────────
	H0 = 0.07159,           # 70 km/s/Mpc in Gyr⁻¹
	rho_crit = 136.0,       # M☉/kpc³
	a_initial = 0.3,        # bounce scale if neg_mass_ratio != 1
	a_final = 1.0,          # a=1 at present day

	# ── Perturbations ────────────────────────────────────────────────────────
	# perturbation_amplitude = δρ/ρ at a=1 (present day).
	# Scaled to a_initial via the anti-universe growth factor D(a).
	perturbation_amplitude = 1.0,   # δρ/ρ ~ 1 at a=1 → nonlinear cosmic web
	anti_correlated = True,         # neg field is inverted: overdensities ↔ underdensities
	spectral_index = 1.0,           # scale-invariant (Harrison-Zel'dovich)

	backreaction = True,
    
	seed = 42,
)

# ── Run simulation ───────────────────────────────────────────────────────────
history = run_cosmological_simulation(config, n_steps=2000, record_every=20)

# ── Visualize ────────────────────────────────────────────────────────────────
ani, fig = animate_cosmological(history, interval=80)

#fig = plot_density_snapshot(history, frame=-1, density_bins=128, slab_frac=0.1)
#fig.savefig('/home/jdb/Pictures/science/cosmo_test.png', dpi=100)
