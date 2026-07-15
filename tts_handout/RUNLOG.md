# Search Pipeline Run Log

## Phase A: Baseline
- **Fitness Design**: Composite including sim_score, rasp_pen, manifold_pen, dur_pen, instability_pen.
- **Listening Observations**: [FILL IN MANUAL NOTES HERE]
- **What changed next**: Selected the top 5 voices for convex blend search.

## Phase B: Convex Blend (Nelder-Mead)
- **Settings**: Softmax weighting of top-5 voices. 30 max iterations.
- **Listening Observations**: [FILL IN MANUAL NOTES HERE]
- **What changed next**: Used this blend as the starting point for SA constrained to PCA subspace.

## Phase C: Constrained Perturbation (Simulated Annealing)
- **Settings**: Base step 0.05, structured per-dim multipliers based on dim_groups.json. PCA K-dim subspace constraint.
- **Listening Observations**: [FILL IN MANUAL NOTES HERE]

