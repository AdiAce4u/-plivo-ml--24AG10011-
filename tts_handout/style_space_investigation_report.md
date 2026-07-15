# Reverse-Engineering the Kokoro Style Latent Space

## Executive Summary
Based on finite-difference testing, the 128/128 split is highly pronounced, but we can empirically verify if there are any leakage dimensions.
**Mantel Correlation (Latent vs Acoustic Distances):** -0.1559

## Step 1 Results (Splicing)
We grafted `af_bella` [0:128] with `bm_george` [128:256].
- **Speaker Similarity to af_bella (Expected High):** 0.9925
- **Speaker Similarity to bm_george (Expected Low):** 0.5446
- **F0 DTW Distance to af_bella (Expected High/Worse):** 601.8052
- **F0 DTW Distance to bm_george (Expected Low/Better):** 6526.6621

## Step 2 Plots/Data (Top Sensitive Dimensions)
**Timbre (0-127) - Top 5 dimensions affecting Speaker Similarity:**
- Dim 113: d_sim = 0.0349
- Dim 10: d_sim = 0.0339
- Dim 120: d_sim = 0.0329
- Dim 49: d_sim = 0.0320
- Dim 59: d_sim = 0.0317

**Prosody (128-255) - Top 5 dimensions affecting Speaker Similarity:**
- Dim 233: d_sim = 0.0358
- Dim 240: d_sim = 0.0326
- Dim 137: d_sim = 0.0319
- Dim 247: d_sim = 0.0318
- Dim 228: d_sim = 0.0306

## Step 3 Mapping (Acoustic Knobs)
- **Dimension 68**: F0 Std (r=-0.42)
- **Dimension 242**: Speaking Rate (r=0.50)
- **Dimension 71**: F1 (r=0.44)
- **Dimension 135**: F2 (r=-0.45)
- **Dimension 209**: F3 (r=-0.41)

## Step 4 Sanity Check
Correlation between Latent L2 Distance and Perceptual Acoustic Distance: **-0.1559**
-> **Recommendation**: Raw L2 distance is poorly correlated with perception. Avoid raw L2 manifold penalties.
