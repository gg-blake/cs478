"""
Name : gpt-shakespeare.py
Description : 
Author : Blake Moody
Date : 9-9-2024
"""
from tokenizer import Tokenizer
import torch
import torch.nn.functional as F

with open("data/d3.txt") as f:
    tokenizer_training_text = f.read()

with open("data/d2.txt") as g:
    model_training_text = g.read()

chars = sorted(list(set(tokenizer_training_text)))

tokenizer = Tokenizer()

#tokenizer.train_stats(tokenizer_training_text, model_training_text, 300)
tokenizer.train(tokenizer_training_text, 300)
data = torch.tensor(tokenizer.encode(model_training_text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]

vocab_size = len(tokenizer._vocab)
validate_data = data[n:]
g = torch.Generator().manual_seed(123456789)
embedding_matrix = torch.rand(vocab_size, vocab_size, generator=g)

batch_size = 8
block_size = 8

def sample(batch_size, block_size):
    starting_indices = torch.randint(len(data) - block_size, (batch_size,))
    sample = torch.stack([data[start_idx:start_idx+block_size] for start_idx in starting_indices])
    target = torch.stack([data[start_idx+1:start_idx+block_size+1] for start_idx in starting_indices])
    return sample, target

s, t = sample(batch_size, block_size)
W = torch.randn((vocab_size, vocab_size), requires_grad=True)

# Forward pass
xenc = F.one_hot(s[0], num_classes=vocab_size).float()
logits = xenc @ W
counts = logits.exp() # Softmax Function
probs = counts / counts.sum(1, keepdim=True) # Softmax Function
loss = -probs[torch.arange(block_size), t[0]].log().mean() # Negative log likelihood
print(loss)

# Backward pass
W.grad = None # Set the gradient to zero
loss.backward()
print(W.grad)






