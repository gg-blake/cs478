from tokenizer import Tokenizer
import numpy as np
import os
import torch
import networkx as nx
import torch.nn.functional as F
from torch import nn
import json
from torch.optim import adamw

device = "cuda" if torch.cuda.is_available() else "cpu"

def cross_entropy(y0, x, e):
    loss = 0.
    n_batch, n_class = y0.shape
    # print(n_class)
    for y1, x1 in zip(y0, x):
        class_index = int(x1.item())
        loss = loss + torch.log(torch.exp(y1[class_index])/(torch.exp(y1).sum()))
    loss = - loss/n_batch
    return loss
        

class AttentionHead(nn.Module):
    def __init__(self, embed_size, head_size):
        super().__init__()
        self.embed_size = embed_size
        self.query_weights = nn.Linear(embed_size, head_size, bias=False, device=device)
        self.key_weights = nn.Linear(embed_size, head_size, bias=False, device=device)
        self.value_weights = nn.Linear(embed_size, head_size, bias=False, device=device)

    def forward(self, embeddings):
        # Queries store the information of what other embeddings have in a particular block
        query = self.query_weights(embeddings)
        # Keys store the information that a particular embedding has relative to other embeddings in a block
        key = self.key_weights(embeddings)
        # By multiplying the keys and queries together, we can allow the embeddings to influence the meaning of other embeddings in the block
        # We need to sqrt(embed_size) to ensure the softmax of wei doesn't get to spiky
        wei = query @ key.transpose(-2, -1) * self.embed_size**-0.5 
        # When training a model, we don't want embeddings that are ahead of an embedding in a block to send information to it (its like cheating in a test)
        # So we will apply a mask to wei
        tril = torch.tril(torch.ones(embeddings.shape[1], embeddings.shape[1], device=device))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        # Then we apply a softmax to make the output on interval [0,1)
        wei = torch.softmax(wei, dim=-1)
        # We don't apply the embeddings directly to wei but instead we apply another backpropagatable linear layer to the embeddings (called value) and then apply wei
        value = self.value_weights(embeddings)
        return wei @ value
    
class MultiheadedAttention(nn.Module):
    def __init__(self, embed_size, head_size, head_count):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(embed_size, head_size) for _ in range(head_count)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, head_size, head_count):
        super().__init__()
        self.head = nn.ModuleList([AttentionHead(embed_size, head_size) for _ in range(head_count)])
        self.ffwd = nn.Sequential(
            nn.Linear(embed_size, embed_size, device=device),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.head_forward(x)
        out = self.linear_forward(out)
        return out

    def head_forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
    def linear_forward(self, x):
        return self.ffwd(x)

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, block_size, head_count):
        super().__init__()
        self.block_size = block_size
        self.token_embeddings = nn.Embedding(vocab_size, embedding_size, device=device)
        self.positional_embeddings = nn.Embedding(block_size, embedding_size, device=device)
        self.block = TransformerBlock(embedding_size, embedding_size//head_count, head_count)
        self.lm_head = nn.Linear(embedding_size, vocab_size, bias=False, device=device)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_idx = self.token_embeddings(idx)
        positional_idx = self.positional_embeddings(torch.arange(T, device=device))
        
        logits = token_idx + positional_idx
        logits = self.block(logits)
        logits = self.lm_head(logits)
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def sample(data, batch_size, block_size):
    starting_indices = torch.randint(len(data) - block_size, (batch_size,))
    sample = torch.stack([data[start_idx:start_idx+block_size] for start_idx in starting_indices])
    target = torch.stack([data[start_idx+1:start_idx+block_size+1] for start_idx in starting_indices])
    return sample, target

    
if __name__ == "__main__":
    tokens = []
    with open("encoded_data/umb100k-1.json") as f:
        tokens = torch.tensor(json.load(f)["www.umb.edu-health-services-counseling-center-index.txt"], dtype=torch.long, device=device)

    s, t = sample(tokens, 4, 8)

    tokenizer = Tokenizer()
    tokenizer.load("tokenizer_models/umb100k-1.model")
    vocab_size = len(tokenizer._vocab)

    lm = LanguageModel(vocab_size, 32, 8, 4)
    
    optimizer = adamw.AdamW(lm.parameters(), lr=1e-3)
    for epoch in range(1000):
        optimizer.zero_grad()
        logits, loss = lm(s, t)
        loss.backward()
        optimizer.step()
        print(loss.item())

    

    
    
        
        

    

    
       