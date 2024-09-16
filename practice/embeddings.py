from tokenizer import Tokenizer
import torch
import torch.nn.functional as F

with open("data/d3.txt") as f:
    tokenizer_training_text = f.read()

with open("data/d2.txt") as g:
    model_training_text = g.read()

# Build a tokenizer
tokenizer = Tokenizer()
tokenizer.train(tokenizer_training_text, 3800, verbose=True)
encoded_text = tokenizer.encode(model_training_text)
vocab_length = len(tokenizer._vocab)
tokens = torch.tensor(encoded_text, dtype=torch.long)
embedding_size = 2



def sample(tokens, batch_size, block_size):
    starting_indices = torch.randint(len(tokens) - block_size, (batch_size,))
    example = torch.stack([tokens[start_idx:start_idx+block_size] for start_idx in starting_indices])
    target = torch.stack([tokens[start_idx+1:start_idx+block_size+1] for start_idx in starting_indices])
    return example, target

batch_size = 32 # number of training examples
block_size = 16 # number of tokens per training example
X, Y = sample(tokens, batch_size, block_size) # Sample the example and target

C = torch.randn((vocab_length, embedding_size))
neuron_count = 100 # This is the number of neurons in the first layer; can be arbitrary
weights_1 = torch.randn((block_size * embedding_size, neuron_count))
biases_1 = torch.randn(neuron_count)
weights_2 = torch.randn((neuron_count, vocab_length))
biases_2 = torch.randn(vocab_length)

parameters = [C, weights_1, weights_2, biases_1, biases_2]
parameters_count = sum(p.nelement() for p in parameters)

for p in parameters:
    p.requires_grad = True

logits = torch.tensor(1)
loss = torch.tensor(1)
for i in range(2000):
    embeddings = C[X] # Shape: 8x8x2
    hidden = torch.tanh(embeddings.view(embeddings.shape[0], block_size * embedding_size) @ weights_1 + biases_1)
    logits = hidden @ weights_2 + biases_2
    loss = F.cross_entropy(logits, Y[:, -1])
    for p in parameters:
        p.grad = None
    loss.backward()
    for p in parameters:
        p.data += -0.1 * p.grad

for i in Y:
    print(tokenizer.decode(i.tolist()))
print("-------------------------------")
print(tokenizer.decode(logits.max(1).indices.tolist()))
# concatenate the second and third dimension of this tensor into a 1-dimensional tensor
# My approach:
#embeddings = torch.cat([embeddings[:,i,:] for i in range(block_size)], 1) # Shape: 8x16
# Andrej 1st approach:
#embeddings = torch.cat(torch.unbind(embeddings, 1), 1)
# Andrej 2nd approach (for memory efficiency):
#embeddings = embeddings.view(batch_size, block_size * embedding_size)
