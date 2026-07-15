import os
import json
import time
import torch
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

import starter.synth as synth
import fitness

def load_dim_groups():
    with open('dim_groups.json', 'r') as f:
        return json.load(f)

def run_step_a(voices):
    print("\n--- PHASE A: Baseline ---")
    results = []
    primary_sentence = fitness.PROBE_POOL[0]
    
    for i, (name, tensor) in enumerate(voices.items()):
        f_val, log_entry = fitness.evaluate_fitness(tensor, primary_sentence)
        log_entry['phase'] = 'A'
        log_entry['name'] = name
        results.append((f_val, name, tensor, log_entry))
        print(f"Baseline {name:20s}: fitness = {f_val:.4f} (sim={log_entry['sim_score']:.4f})")
        
    results.sort(key=lambda x: x[0], reverse=True)
    best_f, best_name, _, _ = results[0]
    print(f"Best Baseline: {best_name} with fitness {best_f:.4f}")
    return results

def run_step_b(top_results, voices):
    print("\n--- PHASE B: Convex Blend Search ---")
    # Take top 5
    K = 5
    top_names = [r[1] for r in top_results[:K]]
    top_tensors = [voices[n] for n in top_names] # [510, 1, 256]
    stacked_tensors = torch.stack(top_tensors) # [K, 510, 1, 256]
    
    primary_sentence = fitness.PROBE_POOL[0]
    
    best_f = -float('inf')
    best_tensor = None
    
    eval_count = 0
    def objective(logits):
        nonlocal best_f, best_tensor, eval_count
        weights = softmax(logits)
        weights_t = torch.tensor(weights, dtype=torch.float32).view(K, 1, 1, 1)
        cand_tensor = (stacked_tensors * weights_t).sum(dim=0)
        
        f_val, log_entry = fitness.evaluate_fitness(cand_tensor, primary_sentence)
        log_entry['phase'] = 'B'
        eval_count += 1
        
        if f_val > best_f:
            best_f = f_val
            best_tensor = cand_tensor.clone()
            print(f"Eval {eval_count} [Phase B]: new best fitness {f_val:.4f}")
            
        return -f_val # minimize

    initial_logits = np.zeros(K)
    initial_logits[0] = 2.0 # Favor the top 1
    
    res = minimize(objective, initial_logits, method='Nelder-Mead', options={'maxiter': 30})
    print(f"Phase B best fitness: {best_f:.4f}")
    return best_tensor, best_f

def run_step_c(start_tensor, dim_groups):
    print("\n--- PHASE C: Constrained Perturbation Search (SA) ---")
    primary_sentence = fitness.PROBE_POOL[0]
    
    curr_tensor = start_tensor.clone()
    curr_f, log = fitness.evaluate_fitness(curr_tensor, primary_sentence)
    
    best_tensor = curr_tensor.clone()
    best_f = curr_f
    
    # 256 step multipliers
    step_mult = np.ones(256)
    for i in range(256):
        if str(i) in dim_groups:
            step_mult[i] = dim_groups[str(i)]['step_multiplier']
            
    step_mult = torch.tensor(step_mult, dtype=torch.float32).view(1, 1, 256)
    
    T = 0.1
    T_min = 0.001
    alpha = 0.9
    base_step = 0.05
    
    iters = 100
    accepted = 0
    
    for i in range(iters):
        # Perturb
        noise = torch.randn(1, 1, 256) * base_step * step_mult
        noise = noise.expand(510, 1, 256)
        cand_tensor = curr_tensor + noise
        
        # Manifold constraint: project the mean onto PCA subspace
        t_256 = cand_tensor.mean(dim=0).squeeze(0).numpy()
        t_proj = fitness.STOCK_PCA.transform([t_256])
        t_rec = fitness.STOCK_PCA.inverse_transform(t_proj)[0]
        t_rec_tensor = torch.tensor(t_rec, dtype=torch.float32).view(1, 1, 256).expand(510, 1, 256)
        
        # Use reconstructed to maintain strict manifold adherence
        cand_tensor = t_rec_tensor
        
        f_val, log_entry = fitness.evaluate_fitness(cand_tensor, primary_sentence)
        log_entry['phase'] = 'C'
        
        delta = f_val - curr_f
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            curr_tensor = cand_tensor
            curr_f = f_val
            accepted += 1
            if f_val > best_f:
                best_tensor = cand_tensor.clone()
                best_f = f_val
                print(f"Iter {i:3d} [Phase C]: new best fitness {best_f:.4f} (accepted)")
                
                if accepted % 5 == 0:
                    os.makedirs('candidates', exist_ok=True)
                    wav = synth.synthesize(primary_sentence, best_tensor)
                    sf = __import__('soundfile')
                    sf.write(f"candidates/cand_f{best_f:.2f}.wav", wav, synth.SR)
        
        T = max(T_min, T * alpha)
        
    print(f"Phase C best fitness: {best_f:.4f}")
    return best_tensor, best_f

def main():
    if not os.path.exists('dim_groups.json'):
        print("Run investigate_tensor.py first!")
        return
        
    dim_groups = load_dim_groups()
    fitness.init_globals()
    
    voices = synth.stock_voices()
    
    # Phase A
    results_a = run_step_a(voices)
    
    # Phase B
    best_b, f_b = run_step_b(results_a, voices)
    
    # Phase C
    best_c, f_c = run_step_c(best_b, dim_groups)
    
    # Final check
    print("\n--- DONE ---")
    print(f"Baseline best: {results_a[0][0]:.4f}")
    print(f"Final best:    {f_c:.4f}")
    
    if best_c.shape != (510, 1, 256):
        print(f"Fixing shape from {best_c.shape} to (510, 1, 256)")
        best_c = best_c.expand(510, 1, 256).clone()
        
    torch.save(best_c, 'voice.pt')
    print("Saved voice.pt")

if __name__ == "__main__":
    main()
