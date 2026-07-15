import synth, torch
v = synth.stock_voices()['af_heart']
print(v.shape)
print("allclose 0, 1:", torch.allclose(v[0], v[1]))
print("allclose 0, 509:", torch.allclose(v[0], v[-1]))
