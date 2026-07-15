import os
import glob
import json
import hashlib
import time
import torch
import numpy as np
import librosa
import soundfile as sf
from sklearn.decomposition import PCA

import starter.synth as synth
import starter.similarity as sim

# --- CONSTANTS & WEIGHTS ---
W_SIM = 1.0
W_RASP = 0.4
W_MANIFOLD = 0.2
W_DUR = 0.15
W_STAB = 0.15

CACHE_DICT = {}
LOG_FILE = "fitness_log.jsonl"
REFERENCE_DIR = "reference"

# --- GLOBAL PRECOMPUTES ---
TARGET_CENTROID = None
TARGET_SPEAKING_RATE = None # characters per second
STOCK_PCA = None
K_PCA = 40 # Will be updated dynamically if we want, but let's say 40 from investigation
STOCK_MEAN = None

def init_globals():
    global TARGET_CENTROID, TARGET_SPEAKING_RATE, STOCK_PCA, STOCK_MEAN, K_PCA
    
    if TARGET_CENTROID is not None: return

    # 1. Target Centroid
    TARGET_CENTROID = sim.target_embedding(REFERENCE_DIR)
    
    # 2. Target Speaking Rate
    total_chars = 0
    total_dur = 0.0
    wavs = sorted(glob.glob(os.path.join(REFERENCE_DIR, "*.wav")))
    txts = sorted(glob.glob(os.path.join(REFERENCE_DIR, "*.txt")))
    for w, t in zip(wavs, txts):
        with open(t, 'r') as f:
            text = f.read().strip()
        wav_data, sr = sf.read(w)
        dur = len(wav_data) / sr
        total_chars += len(text)
        total_dur += dur
    
    TARGET_SPEAKING_RATE = total_chars / total_dur if total_dur > 0 else 15.0
    
    # 3. PCA Subspace
    voices = synth.stock_voices()
    tensors_256 = [v.mean(dim=0).squeeze(0).numpy() for v in voices.values()]
    X = np.stack(tensors_256)
    
    pca = PCA()
    pca.fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    K_PCA = np.argmax(cum_var >= 0.95) + 1
    
    STOCK_PCA = PCA(n_components=K_PCA)
    STOCK_PCA.fit(X)
    STOCK_MEAN = X.mean(axis=0)

def hash_tensor_sentence(tensor, sentence):
    m = hashlib.md5()
    m.update(tensor.numpy().tobytes())
    m.update(sentence.encode('utf-8'))
    return m.hexdigest()

def get_audio_and_features(tensor, sentence):
    h = hash_tensor_sentence(tensor, sentence)
    if h in CACHE_DICT:
        return CACHE_DICT[h]
        
    wav = synth.synthesize(sentence, tensor)
    emb = sim.embed_wav_array(wav)
    
    sr = synth.SR
    dur = len(wav) / sr
    
    # DSP: Raspiness proxy
    # Spectral flatness
    flatness = np.mean(librosa.feature.spectral_flatness(y=wav))
    
    # High freq energy (>4kHz)
    S = np.abs(librosa.stft(wav))
    freqs = librosa.fft_frequencies(sr=sr)
    hf_mask = freqs > 4000
    if np.sum(S) > 0:
        hf_ratio = np.sum(S[hf_mask, :]) / np.sum(S)
    else:
        hf_ratio = 0.0
        
    # Pitch jitter (simple proxy: variance of diffs of F0)
    f0, _, _ = librosa.pyin(wav, fmin=65, fmax=2093, sr=sr)
    valid_f0 = f0[~np.isnan(f0)]
    jitter = 0.0
    if len(valid_f0) > 1:
        jitter = np.std(np.diff(valid_f0)) / (np.mean(valid_f0) + 1e-9)
        
    # Clipping
    clipping = np.mean(np.abs(wav) > 0.99)
    
    rasp_score = float(flatness + hf_ratio + jitter + clipping * 10)
    
    res = {
        'wav': wav,
        'emb': emb,
        'dur': dur,
        'rasp': rasp_score
    }
    CACHE_DICT[h] = res
    return res

PROBE_POOL = [
    "The quick brown fox jumps over the lazy dog.",
    "Please confirm your order number after the beep.",
    "I will call you back tomorrow at three thirty.",
    "Artificial intelligence is transforming our world rapidly.",
    "She sells seashells by the seashore every morning."
]
probe_idx = 0

def evaluate_fitness(tensor, primary_sentence):
    global probe_idx
    init_globals()
    
    # 1. Primary evaluation
    res = get_audio_and_features(tensor, primary_sentence)
    sim_score = sim.cosine(res['emb'], TARGET_CENTROID)
    rasp_pen = res['rasp']
    
    # Duration penalty
    expected_dur = len(primary_sentence) / TARGET_SPEAKING_RATE
    dur_pen = abs(res['dur'] - expected_dur) / expected_dur
    
    # 2. Instability penalty
    # Use 2 rotating sentences
    s1 = PROBE_POOL[probe_idx % len(PROBE_POOL)]
    s2 = PROBE_POOL[(probe_idx + 1) % len(PROBE_POOL)]
    probe_idx = (probe_idx + 1) % len(PROBE_POOL)
    
    r1 = get_audio_and_features(tensor, s1)
    r2 = get_audio_and_features(tensor, s2)
    
    sim1 = sim.cosine(r1['emb'], TARGET_CENTROID)
    sim2 = sim.cosine(r2['emb'], TARGET_CENTROID)
    instability_pen = np.std([sim_score, sim1, sim2])
    
    # 3. Off-manifold penalty
    t_256 = tensor.mean(dim=0).squeeze(0).numpy() # [256]
    t_proj = STOCK_PCA.transform([t_256])
    t_rec = STOCK_PCA.inverse_transform(t_proj)[0]
    manifold_pen = np.linalg.norm(t_256 - t_rec)
    
    # Composite
    fitness_val = (
        W_SIM * sim_score
        - W_RASP * rasp_pen
        - W_MANIFOLD * manifold_pen
        - W_DUR * dur_pen
        - W_STAB * instability_pen
    )
    
    log_entry = {
        'timestamp': time.time(),
        'sim_score': float(sim_score),
        'rasp_pen': float(rasp_pen),
        'manifold_pen': float(manifold_pen),
        'dur_pen': float(dur_pen),
        'instability_pen': float(instability_pen),
        'fitness': float(fitness_val)
    }
    
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(log_entry) + "\n")
        
    return fitness_val, log_entry
