import torch
from sru import SRU, SRUCell

# input has length 20, batch size 32 and dimension 128
torch.manual_seed(0)
x = torch.normal(mean=0.0, std=1.0, size=(20, 32, 128), device="cuda")   # N(0,1)

input_size, hidden_size = 128, 128

rnn = SRU(input_size, hidden_size,
    num_layers = 2,          # number of stacking RNN layers
    dropout = 0.0,           # dropout applied between RNN layers
    bidirectional = False,   # bidirectional RNN
    layer_norm = False,      # apply layer normalization on the output of each layer
    highway_bias = -2,        # initial bias of highway gate (<= 0)
)
rnn.cuda()

output_states, c_states = rnn(x)      # forward pass

print("input has NaN:", bool(torch.isnan(x).any()), " Inf:", bool(torch.isinf(x).any()))
print("output NaN:", bool(torch.isnan(output_states).any()))
print("c_states NaN:", bool(torch.isnan(c_states).any()))
