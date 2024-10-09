import torch
import torch.nn as nn
import torch.nn.functional as F

B, T, C = 4, 8, 2 # batch, time, channels
x = torch.randn(B, T, C)
print(x.shape)

# version 1
# We want x[b, t] = mean_{i <= t} x[b, i]
xbow = torch.zeros((B, T, C))  # xbow = x[bag of words]
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]  # (t, C)
        xbow[b, t] = torch.mean(xprev, 0)

# version 2 using dot products of two matrices
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
print(wei)

xbow2 = wei @ x  # (B, T, T) @ (B, T, C) ----> (B, T, C)
print(torch.allclose(xbow, xbow2))

# version 3: use Softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
# where tril (lower triangle of 1s) == 0 (top part), mask it as -inf
wei = wei.masked_fill(tril == 0, float('-inf'))  
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
# print(torch.allclose(xbow, xbow3))

# version 4: self-attention!
B, T, C = 4, 8, 32  # batch, time, channels
x = torch.randn(B, T, C)

# single Head performing self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, head_size=16)
q = query(x) # (B, T, head_size=16)
wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)


tril = torch.tril(torch.ones(T, T))
# wei = torch.zeros((T, T))
# where tril (lower triangle of 1s) == 0 (top part), mask it as -inf
wei = wei.masked_fill(tril == 0, float('-inf'))  
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
#out = wei @ x
print(wei[0])

'''
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
print(f"a = \n{a}\nb = \n{b}\nc = \n{c}")
'''


