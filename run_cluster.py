"""
Entry point for the Phase 2B galaxy cluster simulation.

Tests the anti-universe cavity model at cluster scale:
  - N galaxies placed at random positions throughout the cluster volume
    (uniform random, not on a shell) with minimum pairwise separation enforced
  - Shared neg-mass background filling the cluster_radius sphere
  - Key observable: mean pairwise separation ⟨d_ij⟩ vs time
    Green line   = simulation (tracked via BH positions for stability)
    Yellow dashed = Hubble de Sitter: d₀ × exp(H₀ t), H₀ = 70 km/s/Mpc

Interpretation:
  - Simulation above/below Hubble line: model over/under-predicts expansion
  - neg_mass_ratio controls "dark energy density"; higher = faster expansion
  - galaxy_separation: minimum distance between any two galaxy centers
    (Andromeda is ~750 kpc away — a physically motivated lower bound)
  - n_disk/n_bulge=0: galaxy treated as a single point mass (faster, cleaner)
    All stellar mass is then in the BH; no disk particles to wander.
  - total_star_budget: auto-scale particles with n_galaxies (keep total N fixed)

Run from the project root:
    python run_cluster.py
"""

from simulations.cluster.sim import (
	ClusterConfig,
	run_cluster_simulation,
	print_cluster_diagnostics,
)
from simulations.cluster.visualize import animate_cluster

config = ClusterConfig(
	# Galaxy count — placed randomly throughout the cluster volume
	n_galaxies = 6,

	# For long cosmological runs (>2 Gyr), use point-mass galaxies.
	# Disk/bulge particles orbit at ±5-10 kpc with ~0.15 Gyr periods; once a BH
	# accelerates rapidly from neg-mass pressure, disk particles become unbound
	# → wild trajectories → energy drift can reach 200%+ by 2 Gyr.
	# Point-mass: full galaxy mass in BH, energy drift < 0.005% over 10 Gyr.
	n_disk_per_galaxy  = 0,
	n_bulge_per_galaxy = 0,

	# Minimum pairwise separation between galaxy centers (kpc).
	galaxy_separation = 400.0,   # kpc — minimum inter-galaxy separation
	cluster_radius    = 1000.0,  # kpc — initial elastic boundary sphere

	# Neg-mass background
	n_negative     = 3000,
	#neg_mass_ratio = 7.0 / 3.0,  # 70% negative mass, same as dark energy fraction
	neg_mass_ratio = 1.0,  # 50/50 neg/pos mass — zero-energy universe

	# Integration
	dt        = 0.005,  # Gyr — safe for point-mass galaxies
	softening = 5.0,    # kpc

	# Start galaxies already in Hubble recession so the simulation begins on the
	# Hubble reference line rather than ramping up from rest.
	hubble_flow_ic = True,

	# Comoving boundary: grow the neg-mass sphere to always surround all galaxies.
	# Uses r_max of BH positions (not r_rms) so even the outermost outlier galaxy
	# is guaranteed to be inside by a factor of safety_factor.
	# Shell theorem: neg-mass outside a galaxy's radius contributes ~zero net force,
	# so any galaxy well inside the sphere feels symmetric pressure from all sides
	# — this is the "infinite-universe window" physics.
	# safety_factor=2.0 keeps every galaxy at ≤50% of the boundary radius.
	comoving_boundary      = True,
	comoving_safety_factor = 2.0,

	# Dark energy equation-of-state (CPL) — shapes the Hubble reference curve.
	# ΛCDM:  w0=-1.0, wa=0.0  →  pure de Sitter exponential d₀·exp(H₀·t)
	# DESI DR2 best fits (arXiv:2503.14738, 2025):
	#   Dataset               Signif   w0      wa
	#   DESI+CMB+Pantheon+    2.8σ    -0.838  -0.62   conservative
	#   DESI+CMB+DESY5        4.2σ    -0.752  -0.86   most significant
	#   DESI+CMB (no SNe)     3.1σ    -0.42   -1.75   aggressive / CMB-only
	# Decaying dark energy (wa<0 with w0>-1) causes the reference to fall below
	# ΛCDM at late times — noticeable difference appears after ~10 Gyr.
	w0 = -0.752,   # DESI+CMB+DESY5 4.2σ
	wa = -0.86,    # DESI+CMB+DESY5 4.2σ

    # Best fit for 70% neg mass ratio
	#w0 = -0.92,
	#wa = -0.25,
)

history = run_cluster_simulation(config, n_steps=4000, record_every=40)
print_cluster_diagnostics(history)
ani, zoom = animate_cluster(history, interval=60)

