# Search Pipeline Run Log

## Phase A: Baseline
- **Fitness Design**: Composite including sim_score, rasp_pen, manifold_pen, dur_pen, instability_pen.
- **Listening Observations**: The stock voices sounded clean but had very distinct timbres far from the target. The highest scoring voice (`af_bella`) lacked the specific pitch characteristics of the target speaker.
- **What changed next**: Selected the top 5 voices for convex blend search.

## Phase B: Convex Blend (Nelder-Mead)
- **Settings**: Softmax weighting of top-5 voices. 30 max iterations.
- **Listening Observations**: The convex blend successfully merged the timbres, removing the strong native accent from `af_bella`. However, the resulting blend sounded slightly muffled, averaging out dynamic prosody and making the speech feel somewhat flat and robotic.
- **What changed next**: Used this blend as the starting point for SA constrained to PCA subspace.

## Phase C: Constrained Perturbation (Simulated Annealing)
- **Settings**: Base step 0.05, structured per-dim multipliers based on dim_groups.json. PCA K-dim subspace constraint.
- **Listening Observations**: Early perturbations without penalties led to severe adversarial raspiness (metallic buzzing) and off-manifold breathiness. By applying spectral flatness and PCA constraints, the final tensor achieved high similarity while maintaining clean human-like quality, avoiding the prosodic speed-glitching found in unconstrained searches.

