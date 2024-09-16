import torch

zeros = torch.zeros(3, 2)
print(zeros)

tensor = torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.6, 7.3]])
print(tensor)

dot = torch.tensordot(zeros, tensor)
print(dot)

eye = torch.eye(5)
print(eye)

a = torch.empty((2, 3), dtype=torch.int64)
empty_like = torch.empty_like(a)
print(empty_like)

probabilities = torch.tensor([0.1, 0.9])
samples = torch.multinomial(probabilities, num_samples=10, replacement=True)
print(samples)

tensor_2 = torch.tensor([1, 2, 3, 4])
out = torch.cat((tensor_2, torch.tensor([5])), dim=0)
print(out)

out_2 = torch.tril(torch.ones(5, 5))
print(out_2)

out_3 = torch.zeros(5, 5).masked_fill(torch.tril(torch.ones(5, 5)) == 0, float('-inf'))
print(out_3)

print(torch.exp(out_3))

input_2 = torch.zeros(2, 3, 4)
print(input_2.shape)
out = input_2.transpose(0, 2)
print(out.shape)

tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
tensor3 = torch.tensor([7, 8, 9])

# Stack the tensors along a new dimension
stacked_tensor = torch.stack([tensor1, tensor2, tensor3])
print(stacked_tensor)

import torch.nn as nn

sample = torch.tensor([10.0, 10.0, 10.0])
linear = nn.Linear(3, 3, bias=False)
print(linear(sample))

import torch.nn.functional as F

# Create a tensor
tensor1 = torch.tensor([3.0, 2.0, 3.0])

# Apply softmax using torch.nn.functional.softmax()
softmax_output = F.softmax(tensor1, dim=0)
print(softmax_output)
total = 0
for num in softmax_output:
    total += num

print(total)
