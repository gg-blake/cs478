from tokenizer import Tokenizer
import numpy as np
import os
import torch
from graphviz import Graph
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
        

class Embedding:
    def __init__(self, v_count, p_count):
        self.v_count = v_count
        self.p_count = p_count
        self.data = torch.rand(v_count, p_count, device=device)

    def __call__(self, idx):
        output = self.data[idx]
        output.unsqueeze(0)
        return output
    
    def cosine_similarity(self, i, j):
        assert i < len(self.data) and j < len(self.data)

cosine_similarity = nn.CosineSimilarity(dim=2)
def loss_fn(output, target):
    return -cosine_similarity(output, target).mean()

class NNLM(nn.Module):
    def __init__(self, tokenizer, p_count):
        super().__init__()
        self.embeddings = nn.Embedding(len(tokenizer._vocab), len(tokenizer._vocab), dtype=torch.half, device=device)

    def forward(self, idx, targets=None):
        logits = self.embeddings(idx)
        '''outputs = self.embeddings(targets)

        
        if targets is None:
            loss = None
        else:
            loss = loss_fn(logits, outputs)'''
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
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

        

'''class VectorDB:
    def __init__(self, tokenizer, embedding_size):
        self.tokenizer = tokenizer
        self.embedding_size = embedding_size
        self._vectors = torch.zeros((0, embedding_size))
        self._vectors_index = Graph()

    def train_from_document(self, file_paths):
        cat_text = ""
        for file_path in file_paths:
            with open(file_path) as f:
                cat_text += f.read()

        self.tokenizer.train(cat_text, self.embedding_size)
        for file_path in file_paths:
            with open(file_path) as f:
                text = f.read()
                self.tokenizer.encode(text)
                tokens = torch.tensor(self.tokenizer.encode(text))
                token_freq = self._token_frequency(tokens)
                self.add_vector(token_freq)

    def _token_frequency(self, tokens):
        token_freq = torch.zeros(len(self.tokenizer._vocab))
        for token in tokens:
            token_freq[token] += 1

        return token_freq.softmax(dim=0)
    
    def add_vector(self, vector):
        # Add the vector to the database (vector should be normalized with softmax)
        self._vectors_index.node(str(len(self._vectors)))
        if len(self._vectors) == 0:
            self._vectors = torch.cat((self._vectors, vector.unsqueeze(0)))
            return
        
        similarity_vector = F.cosine_similarity(vector, self._vectors, dim=1)
        for i, similarity in enumerate(similarity_vector):
            self._vectors_index.edge(str(len(self._vectors)), str(i), label=str(similarity.item()))

        self._vectors = torch.cat((self._vectors, vector.unsqueeze(0)))

    def find_similar_text(self, text, number_of_results=5):
        vector = self._token_frequency(torch.tensor(self.tokenizer.encode(text)))

        # Find the most similar vectors in the database
        similarity_vector = F.cosine_similarity(vector, self._vectors, dim=1)
        similarity_vector = similarity_vector.argsort(descending=True)
        return similarity_vector[:number_of_results]
    
    def visualize(self):
        self._vectors_index.render("graph.png")

if __name__ == "__main__":
    tokenizer = Tokenizer()
    db = VectorDB(tokenizer, 3000)
    db.train_from_document(["data/d1.txt", "data/d2.txt", "data/d3.txt"])
    db.visualize()
    search_text = "Financial aid and the bursars office and student loans"
    print(db._vectors)'''

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

    model = NNLM(tokenizer, 200)
    emb = Embedding(len(tokenizer._vocab), 200)
    logits, loss = model.forward(s, t)
    print(logits)
    print(tokenizer.decode([20000, 31024, 1014]))
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(model.generate(idx, 100)[0].tolist())
    print(tokenizer.decode(model.generate(idx, 100)[0].tolist()))

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = None
    for steps in range(1000):
        xb, yb = sample(tokens, 4, 8)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        

    print(loss.item())
 
    
        
        

    

    
       