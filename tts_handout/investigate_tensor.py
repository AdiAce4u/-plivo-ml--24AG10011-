import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import librosa

import starter.synth as synth
import starter.similarity as sim

def pitch_contour_stats(wav, sr=24000):
    duration = len(wav) / sr
    f0, _, _ = librosa.pyin(wav, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    valid_f0 = f0[~np.isnan(f0)]
    pitch_mean = np.mean(valid_f0) if len(valid_f0) > 0 else 0.0
    pitch_std = np.std(valid_f0) if len(valid_f0) > 0 else 0.0
    return duration, pitch_mean, pitch_std

def main():
    print("Loading stock voices...")
    voices = synth.stock_voices()
    names = sorted(list(voices.keys()))
    
    # 1. PCA Investigation
    # Each tensor is [510, 1, 256]. We average over rows to get a (54, 256) matrix
    tensors_256 = []
    for n in names:
        v = voices[n]
        v_256 = v.mean(dim=0).squeeze(0) # [256]
        tensors_256.append(v_256.numpy())
    X = np.stack(tensors_256) # [54, 256]
    
    print(f"Running PCA on matrix of shape {X.shape}...")
    pca = PCA()
    pca.fit(X)
    explained_variance = pca.explained_variance_ratio_
    cum_variance = np.cumsum(explained_variance)
    
    # Find components for 95%
    k_95 = np.argmax(cum_variance >= 0.95) + 1
    print(f"Number of components needed for 95% variance: {k_95}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(cum_variance)+1), cum_variance, marker='o')
    plt.axvline(x=k_95, color='r', linestyle='--', label=f'95% Variance (k={k_95})')
    plt.title('PCA Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True)
    plt.savefig('pca_variance.png')
    print("Saved pca_variance.png")
    
    # 2. Hypothesis Probing
    rep_name = 'af_heart'
    rep_tensor = voices[rep_name]
    text = "The quick brown fox jumps over the lazy dog."
    
    print(f"\nProbing tensor halves on '{rep_name}'...")
    wav_orig = synth.synthesize(text, rep_tensor)
    orig_emb = sim.embed_wav_array(wav_orig)
    orig_dur, orig_pm, orig_ps = pitch_contour_stats(wav_orig)
    
    # Zero out [0:128]
    t_first_half_zero = rep_tensor.clone()
    t_first_half_zero[:, :, :128] = 0.0
    wav_fh = synth.synthesize(text, t_first_half_zero)
    sim_fh = sim.cosine(orig_emb, sim.embed_wav_array(wav_fh))
    dur_fh, pm_fh, ps_fh = pitch_contour_stats(wav_fh)
    
    # Zero out [128:256]
    t_second_half_zero = rep_tensor.clone()
    t_second_half_zero[:, :, 128:] = 0.0
    wav_sh = synth.synthesize(text, t_second_half_zero)
    sim_sh = sim.cosine(orig_emb, sim.embed_wav_array(wav_sh))
    dur_sh, pm_sh, ps_sh = pitch_contour_stats(wav_sh)
    
    print(f"Original: dur={orig_dur:.2f}s, pitch_mean={orig_pm:.1f}, pitch_std={orig_ps:.1f}")
    print(f"Zero [0:128]  : sim={sim_fh:.4f}, dur={dur_fh:.2f}s, pitch_mean={pm_fh:.1f}, pitch_std={ps_fh:.1f}")
    print(f"Zero [128:256]: sim={sim_sh:.4f}, dur={dur_sh:.2f}s, pitch_mean={pm_sh:.1f}, pitch_std={ps_sh:.1f}")
    
    # To make it robust, we also do an 8-dim block scan
    print("\nRunning 8-dim block scan for fine-grained grouping...")
    blocks = []
    for i in range(0, 256, 8):
        t_block = rep_tensor.clone()
        t_block[:, :, i:i+8] += 0.5 # perturbation
        wav_b = synth.synthesize(text, t_block)
        sim_b = sim.cosine(orig_emb, sim.embed_wav_array(wav_b))
        dur_b, pm_b, ps_b = pitch_contour_stats(wav_b)
        
        sim_drop = 1.0 - sim_b
        dur_diff = abs(dur_b - orig_dur)
        pitch_diff = abs(pm_b - orig_pm) + abs(ps_b - orig_ps)
        
        blocks.append({
            'start': i,
            'end': i+8,
            'sim_drop': sim_drop,
            'dur_diff': dur_diff,
            'pitch_diff': pitch_diff
        })
        print(f"Block {i:3d}-{i+8:3d}: sim_drop={sim_drop:.4f}, dur_diff={dur_diff:.2f}s, pitch_diff={pitch_diff:.1f}")
        
    # Analyze and group
    # We define:
    # Timbre leverage: high sim_drop
    # Prosody leverage: high dur_diff or pitch_diff
    sim_drops = np.array([b['sim_drop'] for b in blocks])
    dur_diffs = np.array([b['dur_diff'] for b in blocks])
    pitch_diffs = np.array([b['pitch_diff'] for b in blocks])
    
    sim_thresh = np.percentile(sim_drops, 60) # Top 40% are high sim
    prosody_score = dur_diffs + pitch_diffs * 0.1
    prosody_thresh = np.percentile(prosody_score, 60) # Top 40% are high prosody
    
    dim_groups = {}
    for b, ps in zip(blocks, prosody_score):
        start = b['start']
        end = b['end']
        
        is_timbre = b['sim_drop'] > sim_thresh
        is_prosody = ps > prosody_thresh
        
        if is_timbre and not is_prosody:
            label = "timbre"
            mult = 2.0
        elif is_prosody and not is_timbre:
            label = "prosody"
            mult = 0.5
        elif is_timbre and is_prosody:
            label = "mixed"
            mult = 1.0
        else:
            label = "low_leverage"
            mult = 0.8
            
        for d in range(start, end):
            dim_groups[str(d)] = {
                "label": label,
                "step_multiplier": mult,
                "sim_drop": b['sim_drop'],
                "prosody_score": ps
            }
            
    with open('dim_groups.json', 'w') as f:
        json.dump(dim_groups, f, indent=2)
    print("\nSaved dim_groups.json")

if __name__ == "__main__":
    main()
