# Does torch.FloatTensor(20, 32, 128).cuda() really contain NaNs, and does SRU then output NaN?
# Same code as reproducer.py plus counts of NaN/Inf in x and NaN in the outputs. No nixnan needed.
# (The garbage depends on what the process allocated earlier, so importing sru first matters:
#  a bare torch script that only allocates x saw 0 NaN, this one sees ~350.)
import torch
from sru import SRU

x = torch.FloatTensor(20, 32, 128).cuda()
print("x: NaN =", torch.isnan(x).sum().item(), " Inf =", torch.isinf(x).sum().item(), " of", x.numel())
rnn = SRU(128, 128, num_layers=2, dropout=0.0, bidirectional=False, layer_norm=False, highway_bias=-2).cuda()
o, c = rnn(x)
print("output NaN:", bool(torch.isnan(o).any()), " c_states NaN:", bool(torch.isnan(c).any()))
