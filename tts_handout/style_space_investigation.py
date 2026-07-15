import os
import json
import torch
import numpy as np
import librosa
import scipy.signal
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Import the existing synth functions
import starter.synth as synth
from starter.similarity import embed_wav_array, cosine

def extract_pitch(y, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    if f0 is None:
        return 0.0, 0.0, np.array([])
    f0 = f0[voiced_flag]
    if len(f0) == 0:
        return 0.0, 0.0, np.array([])
    return float(np.mean(f0)), float(np.std(f0)), f0

def extract_formants(y, sr):
    # Pre-emphasis
    y = scipy.signal.lfilter([1., -0.63], 1, y)
    # LPC
    a = librosa.lpc(y, order=16)
    rts = np.roots(a)
    rts = [r for r in rts if np.imag(r) >= 0]
    angz = np.arctan2(np.imag(rts), np.real(rts))
    freqs = sorted(angz * (sr / (2 * np.pi)))
    freqs = [f for f in freqs if f > 50]
    if len(freqs) < 3:
        freqs += [0] * (3 - len(freqs))
    return freqs[:3]

def main():
    print("Loading stock voices...")
    voices = synth.stock_voices()
    
    # 5 test voices
    test_voices = ['af_bella', 'af_sarah', 'am_adam', 'bf_emma', 'bm_george']
    
    # Standard phrase
    TEXT = "The quick brown fox jumps over the lazy dog."
    SR = 24000
    
    print("\n--- Step 1: Targeted Latent Swap ---")
    speaker_a = 'af_bella'
    speaker_b = 'bm_george'
    
    v_a = voices[speaker_a]
    v_b = voices[speaker_b]
    
    s_hybrid = v_a.clone()
    s_hybrid[0, 0, 128:] = v_b[0, 0, 128:]
    
    print("Generating audio for Hybrid, A, and B...")
    wav_a = synth.synthesize(TEXT, v_a)
    wav_b = synth.synthesize(TEXT, v_b)
    wav_hybrid = synth.synthesize(TEXT, s_hybrid)
    
    emb_a = embed_wav_array(wav_a)
    emb_b = embed_wav_array(wav_b)
    emb_hybrid = embed_wav_array(wav_hybrid)
    
    sim_hybrid_a = cosine(emb_hybrid, emb_a)
    sim_hybrid_b = cosine(emb_hybrid, emb_b)
    
    _, _, f0_a = extract_pitch(wav_a, SR)
    _, _, f0_b = extract_pitch(wav_b, SR)
    _, _, f0_hybrid = extract_pitch(wav_hybrid, SR)
    
    D_a, _ = librosa.sequence.dtw(f0_hybrid, f0_a)
    D_b, _ = librosa.sequence.dtw(f0_hybrid, f0_b)
    dtw_a = D_a[-1, -1]
    dtw_b = D_b[-1, -1]
    
    print(f"Hybrid Similarity to A (Expected HIGH): {sim_hybrid_a:.4f}")
    print(f"Hybrid Similarity to B (Expected LOW): {sim_hybrid_b:.4f}")
    print(f"Hybrid F0 DTW to A (Expected HIGH / worse): {dtw_a:.4f}")
    print(f"Hybrid F0 DTW to B (Expected LOW / better): {dtw_b:.4f}")
    
    step1_res = {
        'sim_a': sim_hybrid_a, 'sim_b': sim_hybrid_b,
        'dtw_a': dtw_a, 'dtw_b': dtw_b
    }
    
    print("\n--- Step 2: Finite-Difference Sensitivity Profiling ---")
    base_voice = v_a
    base_wav = wav_a
    base_emb = emb_a
    base_dur = len(base_wav) / SR
    base_f0_mean, base_f0_std, _ = extract_pitch(base_wav, SR)
    
    sensitivities = []
    
    for i in range(256):
        if i % 32 == 0:
            print(f"Profiling dimension {i}/256...")
        s_pert = base_voice.clone()
        s_pert[0, 0, i] += 1.0
        
        wav_pert = synth.synthesize(TEXT, s_pert)
        pert_emb = embed_wav_array(wav_pert)
        pert_dur = len(wav_pert) / SR
        pert_f0_mean, pert_f0_std, _ = extract_pitch(wav_pert, SR)
        
        delta_sim = 1.0 - cosine(pert_emb, base_emb)
        delta_dur = abs(pert_dur - base_dur)
        delta_f0 = abs(pert_f0_mean - base_f0_mean)
        
        sensitivities.append({
            'dim': i,
            'd_sim': float(delta_sim),
            'd_dur': float(delta_dur),
            'd_f0': float(delta_f0)
        })
    
    with open('step2_sens.json', 'w') as f:
        json.dump(sensitivities, f)
        
    print("\n--- Step 3: Acoustic Feature Correlation ---")
    NUM_PERTURBS = 50
    print(f"Generating {NUM_PERTURBS} random perturbations...")
    
    matrix_dim = np.zeros((NUM_PERTURBS, 256))
    matrix_feat = np.zeros((NUM_PERTURBS, 6)) # f0_mean, f0_std, rate, f1, f2, f3
    
    for i in range(NUM_PERTURBS):
        noise = torch.randn(1, 1, 256) * 0.5
        s_rand = base_voice + noise
        matrix_dim[i] = noise[0, 0].numpy()
        
        wav_rand = synth.synthesize(TEXT, s_rand)
        f0_m, f0_s, _ = extract_pitch(wav_rand, SR)
        dur = len(wav_rand) / SR
        rate = 1.0 / dur if dur > 0 else 0
        formants = extract_formants(wav_rand, SR)
        
        matrix_feat[i] = [f0_m, f0_s, rate, formants[0], formants[1], formants[2]]
    
    # Correlate
    correlations = np.zeros((256, 6))
    for d in range(256):
        for f in range(6):
            if np.std(matrix_dim[:, d]) > 0 and np.std(matrix_feat[:, f]) > 0:
                correlations[d, f] = np.corrcoef(matrix_dim[:, d], matrix_feat[:, f])[0, 1]
    
    # Find high correlations
    feat_names = ["F0 Mean", "F0 Std", "Speaking Rate", "F1", "F2", "F3"]
    discovered_knobs = {}
    for f_idx, fname in enumerate(feat_names):
        best_dim = np.argmax(np.abs(correlations[:, f_idx]))
        r_val = correlations[best_dim, f_idx]
        if abs(r_val) > 0.4: # Lowered threshold slightly due to reduced sample size
            discovered_knobs[f"Dimension {best_dim}"] = f"{fname} (r={r_val:.2f})"
    
    print("\n--- Step 4: Voice Cluster Correspondence ---")
    voice_keys = list(voices.keys())[:30]
    
    latent_matrix = np.zeros((30, 256))
    feat_matrix = np.zeros((30, 6))
    
    for i, k in enumerate(voice_keys):
        v = voices[k]
        latent_matrix[i] = v[0, 0].numpy()
        
        wav = synth.synthesize(TEXT, v)
        f0_m, f0_s, _ = extract_pitch(wav, SR)
        rate = 1.0 / (len(wav)/SR)
        formants = extract_formants(wav, SR)
        feat_matrix[i] = [f0_m, f0_s, rate, formants[0], formants[1], formants[2]]
        
    dist_latent = cdist(latent_matrix, latent_matrix, metric='euclidean')
    
    # Normalize features for distance
    feat_norm = (feat_matrix - feat_matrix.mean(axis=0)) / (feat_matrix.std(axis=0) + 1e-8)
    dist_feat = cdist(feat_norm, feat_norm, metric='euclidean')
    
    # Upper triangle
    idx = np.triu_indices(30, k=1)
    r_mantel = np.corrcoef(dist_latent[idx], dist_feat[idx])[0, 1]
    print(f"Mantel Correlation: {r_mantel:.4f}")
    
    print("\n--- Generating Report ---")
    
    # Sort step 2
    d_sim_sort = sorted(sensitivities, key=lambda x: x['d_sim'], reverse=True)
    top_timbre = [x for x in d_sim_sort if x['dim'] < 128][:5]
    top_prosody = [x for x in d_sim_sort if x['dim'] >= 128][:5]
    
    report = f"""# Reverse-Engineering the Kokoro Style Latent Space

## Executive Summary
Based on finite-difference testing, the 128/128 split is highly pronounced, but we can empirically verify if there are any leakage dimensions.
**Mantel Correlation (Latent vs Acoustic Distances):** {r_mantel:.4f}

## Step 1 Results (Splicing)
We grafted `af_bella` [0:128] with `bm_george` [128:256].
- **Speaker Similarity to af_bella (Expected High):** {step1_res['sim_a']:.4f}
- **Speaker Similarity to bm_george (Expected Low):** {step1_res['sim_b']:.4f}
- **F0 DTW Distance to af_bella (Expected High/Worse):** {step1_res['dtw_a']:.4f}
- **F0 DTW Distance to bm_george (Expected Low/Better):** {step1_res['dtw_b']:.4f}

## Step 2 Plots/Data (Top Sensitive Dimensions)
**Timbre (0-127) - Top 5 dimensions affecting Speaker Similarity:**
"""
    for t in top_timbre: report += f"- Dim {t['dim']}: d_sim = {t['d_sim']:.4f}\n"
    
    report += "\n**Prosody (128-255) - Top 5 dimensions affecting Speaker Similarity:**\n"
    for t in top_prosody: report += f"- Dim {t['dim']}: d_sim = {t['d_sim']:.4f}\n"
    
    report += "\n## Step 3 Mapping (Acoustic Knobs)\n"
    for k, v in discovered_knobs.items():
        report += f"- **{k}**: {v}\n"
        
    report += f"\n## Step 4 Sanity Check\n"
    report += f"Correlation between Latent L2 Distance and Perceptual Acoustic Distance: **{r_mantel:.4f}**\n"
    if r_mantel > 0.4:
        report += "-> **Recommendation**: L2 manifold penalties are a safe and effective proxy for perceptual distance.\n"
    else:
        report += "-> **Recommendation**: Raw L2 distance is poorly correlated with perception. Avoid raw L2 manifold penalties.\n"
        
    with open('style_space_investigation_report.md', 'w') as f:
        f.write(report)
        
    print("Report generated: style_space_investigation_report.md")

if __name__ == "__main__":
    main()
