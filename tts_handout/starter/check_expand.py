import synth, torch
import similarity as sim
v = synth.stock_voices()['af_heart']
v_mean = v.mean(dim=0, keepdim=True).expand(510, 1, 256).clone()

text = "The quick brown fox jumps over the lazy dog."
wav_orig = synth.synthesize(text, v)
wav_mean = synth.synthesize(text, v_mean)

emb_orig = sim.embed_wav_array(wav_orig)
emb_mean = sim.embed_wav_array(wav_mean)
similarity = sim.cosine(emb_orig, emb_mean)
print("Similarity between orig and mean-expanded:", similarity)
