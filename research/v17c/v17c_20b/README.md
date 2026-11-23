# V17C.20B

Independent particle-parameter, Coulomb, perturbation and basin study for `refine12_sobol17`, `Local15` and `candidate52`.

Start with [the Russian report](PARTICLE_ATTRACTOR_SEARCH_RU.md). The parameter and attractor registries, local-stability audit and session state are kept beside it. All generated numerical artifacts, frozen start states and resumable checkpoints are in `results/`; publication figures and scripts are in `figures/` and `code/`.

The numerical verdict is deliberately limited: long-lived counterrotation is reproduced, including a 240,000-period replay and an independent DOP853 check. No attractor or basin probability is claimed.

Run instructions are in [SESSION_HANDOFF.md](SESSION_HANDOFF.md). File SHA-256 digests are listed in [MANIFEST.json](MANIFEST.json).
