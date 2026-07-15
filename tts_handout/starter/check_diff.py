import synth, torch
v = synth.stock_voices()['af_heart']
print("max diff 0,1:", torch.max(torch.abs(v[0] - v[1])).item())
print("mean diff 0,1:", torch.mean(torch.abs(v[0] - v[1])).item())
print("std of v[0]:", torch.std(v[0]).item())
