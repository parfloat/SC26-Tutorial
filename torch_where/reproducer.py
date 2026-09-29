import torch

device = torch.device("cuda")
a = torch.tensor(100., requires_grad=True, device=device)
b = torch.where(a <= 0, torch.exp(a), 1 + a)
b.backward()

print(b)
print(a.grad)
