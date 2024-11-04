"""
Name : lm_model.py
Description : Generates text using a transformer model trained on a dataset
Author : Blake Moody
Date : 10-18-2024
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import adamw
import tiktoken
from math import floor
from tqdm import tqdm
import argparse
import optparse
from model_config import *
import os
import datasets
import numpy as np
import torch
from transformers import BertTokenizer
import nltk
from nltk.tokenize import sent_tokenize
import transformers
import random
import math
from datasets import Features, Sequence, Value
import pyarrow as pa

nltk.download('punkt_tab')  # Download the Punkt sentence tokenizer if not already present

# Load the default model configuration
LM_MODEL_CONFIG = [
    EMBEDDING_SIZE,
    BATCH_SIZE,
    BLOCK_SIZE,
    LEARNING_RATE,
    STEPS,
    HEAD_COUNT,
    LAYER_COUNT,
    DROPOUT
]

# Set the device to use for training
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("Warning: Using CPU for training; consider using a GPU for faster training")

class AttentionHead(nn.Module):
    """
    A class that represents a single attention head of the transformer model architecture;
    The attention head is used to calculate the attention scores of each node in a block of tokens;
    The structure of the class mirrors the architecture specified in the Attention is All You Need paper (https://arxiv.org/abs/1706.03762)

    Attributes
    ----------
    embed_size : int
        The number of embedding dimensions
    head_size : int
        An arbitrary shared size for the query, key, and value weights
    block_size : int
        The number of tokens in a block
    dropout : nn.Dropout
        The dropout layer
    query_weights : nn.Linear
        The linear layer for the query weights
    key_weights : nn.Linear
        The linear layer for the key weights
    value_weights : nn.Linear
        The linear layer for the value weights
    """
    def __init__(self, embed_size, head_size, block_size, dropout):
        """
        Parameters
        ----------
        embed_size : int
            The number of embedding dimensions
        head_size : int
            An arbitrary shared size for the query, key, and value weights
        block_size : int
            The number of tokens in a block
        dropout : float
            The rate at which nodes in the network are randomly zeroed out during training to prevent overfitting
        """
        super().__init__()
        self.embed_size = embed_size
        self.query_weights = nn.Linear(embed_size, head_size, bias=False, device=device)
        self.key_weights = nn.Linear(embed_size, head_size, bias=False, device=device)
        self.value_weights = nn.Linear(embed_size, head_size, bias=False, device=device)
        self.dropout = nn.Dropout(dropout)
        # We want to apply a mask to the attention scores to prevent the model from cheating during training
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, device=device))) # Lower triangular matrix

    def forward(self, embeddings, masked=True):
        """
        Forward passes a list of embeddings through the attention head and returns the attention scores

        Parameters
        ----------
        embeddings : torch.Tensor
            The embeddings [B x T x C] where B is the batch size, T is the number of tokens in a block, and C is the number of embedding dimensions
        """
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
        if masked:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # Then we apply a softmax to make the output on interval [0,1)
        wei = torch.softmax(wei, dim=-1)
        # We don't apply the embeddings directly to wei but instead we apply another backpropagatable linear layer to the embeddings (called value) and then apply wei
        value = self.value_weights(embeddings)
        return wei @ value


class TransformerBlock(nn.Module):
    """
    A class that represents a transformer block that can be used in a transformer model;
    The transformer block consists of a multiheaded attention layer and a feed forward layer;
    The multiheaded attention layer is used to calculate the attention scores of each node in a block of tokens;
    The feed forward layer is used to train the nodes to compute their attention scores individually;
    The structure of the class mirrors the architecture specified in the Attention is All You Need paper (https://arxiv.org/abs/1706.03762)

    Attributes
    ----------
    heads : nn.ModuleList
        The multiheaded attention layers
    proj : nn.Linear
        The linear projection of the outcome of the multiheaded attention layer
    dropout : nn.Dropout
        The dropout layer
    ffwd : nn.Sequential
        The feed forward layer
    layer_norm1 : nn.LayerNorm
        The layer normalization layer for the multiheaded attention layer
    layer_norm2 : nn.LayerNorm
        The layer normalization layer for the feed forward layer
    """
    def __init__(self, embed_size, head_size, head_count, block_size, dropout):
        """
        Parameters
        ----------
        embed_size : int
            The size of the embeddings
        head_size : int
            The size of the heads in the multiheaded attention layer
        head_count : int
            The number of heads in the multiheaded attention layer
        block_size : int
            The number of tokens in a block
        dropout : float
            The rate at which nodes in the network are randomly zeroed out during training to prevent overfitting
        """
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
        """
        Forward pass of the model of a block of tokens; each block consists of a number of tokens from the training/validation data

        Parameters
        ----------
        x : torch.Tensor
            The block of tokens [B x T x C] where B is the batch size, T is the number of tokens in a block, and C is the number of embedding dimensions
        """
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
        """
        Helper function that forward passes the data through the multiheaded attention layer

        Parameters
        ----------
        x : torch.Tensor
            The block of tokens [B x T x C] where B is the batch size, T is the number of tokens in a block, and C is the number of embedding dimensions
        """
        # We want to apply the multiheaded attention layer to the data so concatenate the outcomes of each head
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        # We want to recombine the outcomes together so we must project it to a layer of the right dimensions 
        # (head_count x embed_size x [embed_size // head_count]) -> (embed_size x embed_size)
        out = self.dropout(out)
        out = self.proj(out)
        return out
    
    def linear_forward(self, x):
        """
        Helper function that forward passes the data through the feed forward layer

        Parameters
        ----------
        x : torch.Tensor
            The block of tokens [B x T x C] where B is the batch size, T is the number of tokens in a block, and C is the number of embedding dimensions
        """
        return self.ffwd(x)


class LanguageModel(nn.Module):
    """
    A class that represents a language model that can be trained on a dataset and generate text

    Attributes
    ----------
    batch_size : int
        The number of samples to process in a single forward pass
    block_size : int
        The number of tokens in a block
    learning_rate : float
        The learning rate for the optimizer
    steps : int
        The number of steps to train the model
    token_embeddings : nn.Embedding
        The embeddings for the tokens
    positional_embeddings : nn.Embedding
        The embeddings for the positions of the tokens
    blocks : nn.Sequential
        The transformer blocks
    layer_norm : nn.LayerNorm
        The layer normalization layer
    lm_head : nn.Linear
        The linear layer for the language model head

    Methods
    -------
    forward(idx, targets=None)
        Forward pass of the model
    generate(idx, max_new_tokens)
        Generate text from the model
    train_model(tokens, eval_iters=200, training_val_ratio=0.8, loss_report_interval=500)
        Train the model on a dataset
    _estimate_loss(eval_iters, training_data, validation_data)
        Estimate the loss of the model on a dataset
    """
    def __init__(self, vocab_size, embedding_size, batch_size, block_size, learning_rate, steps, head_count, layer_count, dropout):
        """
        Parameters
        ----------
        vocab_size : int
            The size of the vocabulary
        embedding_size : int
            The size of the embeddings
        batch_size : int
            The number of samples to process in a single forward pass
        block_size : int
            The number of tokens in a block
        learning_rate : float
            The learning rate for the optimizer
        steps : int
            The number of steps to train the model
        head_count : int
            The number of heads in the multiheaded attention layer
        layer_count : int
            The number of transformer blocks
        dropout : float
            The rate at which nodes in the network are randomly zeroed out during training to prevent overfitting
        """
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
        self.current_index = 0

    def forward(self, idx, targets=None):
        """
        Forward pass of the model of a batch of tokens; each batch consistss of a number of blocks/examples of tokens from the training/validation data

        Parameters
        ----------
        idx : torch.Tensor
            The batch of tokens [B x T] where B is the batch size and T is the number of tokens in a block
        targets : torch.Tensor, optional
            The target tokens [B x T]; this is normally the idx tensor shifted by one token to the right in all the batches to predict the next token; parameter is only specified during training
        """
        B, T = idx.shape
        token_idx = self.token_embeddings(idx)
        positional_idx = self.positional_embeddings(torch.arange(T, device=device))
        
        logits = token_idx + positional_idx
        logits = self.blocks(logits)
        logits = self.layer_norm(logits)
        logits = self.lm_head(logits)
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss
    
class BERT(nn.Module):
    """
    A class that represents a language model that can be trained on a dataset and generate text

    Attributes
    ----------
    batch_size : int
        The number of samples to process in a single forward pass
    block_size : int
        The number of tokens in a block
    learning_rate : float
        The learning rate for the optimizer
    steps : int
        The number of steps to train the model
    token_embeddings : nn.Embedding
        The embeddings for the tokens
    positional_embeddings : nn.Embedding
        The embeddings for the positions of the tokens
    blocks : nn.Sequential
        The transformer blocks
    layer_norm : nn.LayerNorm
        The layer normalization layer
    lm_head : nn.Linear
        The linear layer for the language model head

    Methods
    -------
    forward(idx, targets=None)
        Forward pass of the model
    generate(idx, max_new_tokens)
        Generate text from the model
    train_model(tokens, eval_iters=200, training_val_ratio=0.8, loss_report_interval=500)
        Train the model on a dataset
    _estimate_loss(eval_iters, training_data, validation_data)
        Estimate the loss of the model on a dataset
    """
    def __init__(self, vocab_size, embedding_size, batch_size, block_size, learning_rate, steps, head_count, layer_count, dropout):
        """
        Parameters
        ----------
        vocab_size : int
            The size of the vocabulary
        embedding_size : int
            The size of the embeddings
        batch_size : int
            The number of samples to process in a single forward pass
        block_size : int
            The number of tokens in a block
        learning_rate : float
            The learning rate for the optimizer
        steps : int
            The number of steps to train the model
        head_count : int
            The number of heads in the multiheaded attention layer
        layer_count : int
            The number of transformer blocks
        dropout : float
            The rate at which nodes in the network are randomly zeroed out during training to prevent overfitting
        """
        super().__init__()
        self.batch_size = batch_size
        self.block_size = block_size
        self.learning_rate = learning_rate
        self.steps = steps
        self.vocab_size = vocab_size
        self.token_embeddings = nn.Embedding(vocab_size+2, embedding_size, device=device)
        self.segment_embeddings = nn.Embedding(2, embedding_size, device=device)
        self.positional_embeddings = nn.Embedding(block_size, embedding_size, device=device)
        self.blocks = nn.Sequential(*[TransformerBlock(embedding_size, embedding_size, head_count, block_size, dropout) for _ in range(layer_count)])
        self.layer_norm = nn.LayerNorm(embedding_size, device=device)
        self.lm_head = nn.Linear(embedding_size, vocab_size, bias=False, device=device)
        self.current_index = 0

    def forward(self, input_ids, attention_mask, sentence_ids, targets=None):
        """
        Forward pass of the model of a batch of tokens; each batch consistss of a number of blocks/examples of tokens from the training/validation data

        Parameters
        ----------
        idx : torch.Tensor
            The batch of tokens [B x T] where B is the batch size and T is the number of tokens in a block
        targets : torch.Tensor, optional
            The target tokens [B x T]; this is normally the idx tensor shifted by one token to the right in all the batches to predict the next token; parameter is only specified during training
        """
        B, T = idx.shape
        
        B, T = idx.shape
        token_idx = self.token_embeddings(idx)
        positional_idx = self.positional_embeddings(torch.arange(T, device=device))
        sentence_idx = self.sentence_embeddings()
        logits = token_idx + positional_idx
        logits = self.blocks(logits)
        logits = self.layer_norm(logits)
        logits = self.lm_head(logits)
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss





if __name__ == "__main__":
    

    # 256 * 4bytes * 3 (attention mask, input ids, and token type ids) * 8013769 examples = ~24.6GB

    
        

