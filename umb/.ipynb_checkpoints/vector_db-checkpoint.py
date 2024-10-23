from tokenizer import Tokenizer
import numpy as np
import os
import torch
import networkx as nx
import torch.nn.functional as F
from torch import nn
import json
from torch.optim import adamw
import tiktoken
from math import floor
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"



class AttentionHead(nn.Module):
    def __init__(self, embed_size, head_size, block_size, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.query_weights = nn.Linear(embed_size, head_size, bias=False, device=device)
        self.key_weights = nn.Linear(embed_size, head_size, bias=False, device=device)
        self.value_weights = nn.Linear(embed_size, head_size, bias=False, device=device)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, device=device)))

    def forward(self, embeddings):
        B, T, C = embeddings.shape
        # Queries store the information of what other embeddings have in a particular block
        query = self.query_weights(embeddings)
        # Keys store the information that a particular embedding has relative to other embeddings in a block
        key = self.key_weights(embeddings)
        # By multiplying the keys and queries together, we can allow the embeddings to influence the meaning of other embeddings in the block
        # We need to sqrt(embed_size) to ensure the softmax of wei doesn't get to spiky
        wei = query @ key.transpose(-2, -1) * self.embed_size**-0.5 
        # When training a model, we don't want embeddings that are ahead of an embedding in a block to send information to it (its like cheating in a test)
        # So we will apply a mask to wei
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # Then we apply a softmax to make the output on interval [0,1)
        wei = torch.softmax(wei, dim=-1)
        # We don't apply the embeddings directly to wei but instead we apply another backpropagatable linear layer to the embeddings (called value) and then apply wei
        value = self.value_weights(embeddings)
        return wei @ value
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, head_size, head_count, block_size, dropout):
        super().__init__()
        # Multiheaded attention (batched attention calculation)
        self.heads = nn.ModuleList([AttentionHead(embed_size, head_size // head_count, block_size, dropout) for _ in range(head_count)])
        # Linear projection of outcome of multiheaded attention layer
        self.proj = nn.Linear(embed_size, embed_size, device=device)
        # Randomly zeros out some of the data to prevent overfitting in training
        self.dropout = nn.Dropout(dropout)
        # Simple multilayered perceptron
        self.ffwd = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size, device=device),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size, device=device),
            self.dropout
        )
        self.layer_norm1 = nn.LayerNorm(embed_size, device=device)
        self.layer_norm2 = nn.LayerNorm(embed_size, device=device)

    def forward(self, x):
        # We want to ensure that our nodes across each batch dimension have mean = 0 and standard deviation = 0 before feeding to the multiheaded attention layer
        # So we want to apply whats called layer normalization
        # Here is the pytorch documentation: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html (LayerNorm)
        layer_norm = self.layer_norm1(x)
        # Both the multiheaded attention layer and feed forward layer add the in features of the layer to the out features
        # This is what is referred to as residual connections, and it solves an issue where increasingly deep networks become hard to train/optimize
        # The paper discussing the benefits of this can be found here: https://arxiv.org/abs/1512.03385 (Deep Residual Learning for Image Recognition)
        x = x + self.head_forward(layer_norm)
        # We also want to apply layer normalization to our attention output before passing it to the feed forward layer
        # In the original Attention is All You Need paper, layer normalization comes after each layer, but better results come from doing pre-layer normalization
        layer_norm = self.layer_norm2(x)
        # Once all the nodes in the head have their individual attention scores, we need to train the nodes to compute their attention scores individually
        # This is why we feed the data into a multilayered perceptron, which will allow the model to recognize patterns in the data
        x = x + self.linear_forward(layer_norm)
        return x

    def head_forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        # We want to recombine the outcomes together so we must project it to a layer of the right dimensions 
        # (head_count x embed_size x [embed_size // head_count]) -> (embed_size x embed_size)
        out = self.dropout(out)
        out = self.proj(out)
        return out
    
    def linear_forward(self, x):
        return self.ffwd(x)
    
    

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, batch_size, block_size, learning_rate, steps, head_count, layer_count, dropout):
        super().__init__()
        self.batch_size = batch_size
        self.block_size = block_size
        self.learning_rate = learning_rate
        self.steps = steps
        self.token_embeddings = nn.Embedding(vocab_size, embedding_size, device=device)
        self.positional_embeddings = nn.Embedding(block_size, embedding_size, device=device)
        self.blocks = nn.Sequential(*[TransformerBlock(embedding_size, embedding_size, head_count, block_size, dropout) for _ in range(layer_count)])
        self.layer_norm = nn.LayerNorm(embedding_size, device=device)
        self.lm_head = nn.Linear(embedding_size, vocab_size, bias=False, device=device)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_idx = self.token_embeddings(idx)
        positional_idx = self.positional_embeddings(torch.arange(T, device=device))
        
        logits = token_idx + positional_idx
        logits = self.blocks(logits)
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
    
    def train_model(self, tokens, eval_iters=200, training_val_ratio=0.8, loss_report_interval=500):
        training_tokens = tokens[:floor(len(tokens)*training_val_ratio)]
        validation_tokens = tokens[floor(len(tokens)*training_val_ratio):]
        optimizer = adamw.AdamW(self.parameters(), lr=self.learning_rate)
        for step in tqdm(range(self.steps)):
            optimizer.zero_grad()
            s, t = sample(training_tokens, 4, 8)
            logits, loss = lm(s, t)
            loss.backward()
            optimizer.step()
            if step % loss_report_interval == 0:
                losses = self.estimate_loss(eval_iters, training_tokens, validation_tokens)
                print(f"step {step}: train loss {losses[0]:.4f}, val loss {losses[1]:.4f}")
                
                #print(f"step {step}: train loss {loss:.4f}, val loss {loss:.4f}")

    @torch.no_grad()
    def estimate_loss(self, eval_iters, training_data, validation_data):
        out = {}
        # Disable dropout and layer normalization before model validation
        self.eval()
        for i, split in enumerate([training_data, validation_data]):
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = sample(split, self.batch_size, self.block_size)
                logits, loss = self(X, Y)
                losses[k] = loss.item()
            out[i] = losses.mean()
        # Enable dropout and layer normalization after model validation
        self.train()
        return out


def sample(data, batch_size, block_size):
    starting_indices = torch.randint(len(data) - block_size, (batch_size,))
    sample = torch.stack([data[start_idx:start_idx+block_size] for start_idx in starting_indices])
    target = torch.stack([data[start_idx+1:start_idx+block_size+1] for start_idx in starting_indices])
    return sample, target

def useTiktoken(filename):
    tokenizer = tiktoken.get_encoding("o200k_base")
    assert tokenizer.decode(tokenizer.encode("hello world")) == "hello world"
    with open(filename) as f:
        tokens = torch.tensor(tokenizer.encode(f.read()), dtype=torch.long, device=device)

    return tokenizer, tokens, tokenizer.n_vocab

def useLocal(filename):
    tokenizer = Tokenizer()
    tokenizer.load("tokenizer_models/umb100k-1.model")
    assert tokenizer.decode(tokenizer.encode("hello world")) == "hello world"
    with open(filename) as f:
        tokens = torch.tensor(tokenizer.encode(f.read()), dtype=torch.long, device=device)

    return tokenizer, tokens, len(tokenizer._vocab)

    
if __name__ == "__main__":
    tokens = []
    '''with open("encoded_data/umb100k-1.json") as f:
        tokens = torch.tensor(json.load(f)["www.umb.edu-health-services-counseling-center-index.txt"], dtype=torch.long, device=device)'''

    tokenizer, tokens, vocab_size = useTiktoken("data/threebody.txt")
    
    '''tokenizer.load("tokenizer_models/umb100k-1.model")
    vocab_size = len(tokenizer._vocab)'''
    

    lm = LanguageModel(
        vocab_size=vocab_size, 
        embedding_size=32,
        batch_size=16, 
        block_size=256,
        learning_rate=3e-4,
        steps=3000, 
        head_count=4, 
        layer_count=3,
        dropout=0.2
        )
    

    lm.train_model(tokens)
    start_idx, _ = sample(tokens, lm.batch_size, lm.block_size)
    outputs = lm.generate(start_idx, max_new_tokens=400)[0].tolist()
    print(f"Prompt:\n{tokenizer.decode(start_idx[0].tolist())}\nGenerated Response:\n{tokenizer.decode(outputs)}")
    
    
    '''optimizer = adamw.AdamW(lm.parameters(), lr=1e-3)
    training_val_ratio = 0.7
    training_tokens = tokens[:floor(len(tokens)*training_val_ratio)]
    validation_tokens = tokens[floor(len(tokens)*training_val_ratio):]
    for epoch in range(2000):
        optimizer.zero_grad()
        s, t = sample(tokens, 4, 8)
        logits, loss = lm(s, t)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch} Loss: {loss}")'''

    

    
    
        
        

    

    
       