from tokenizer import Tokenizer
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import adamw
import tiktoken
from math import floor
from tqdm import tqdm
import argparse
import optparse

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

    def forward(self, embeddings):
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
        """
        Generate text from the model given an initial set of sample tokens; it's essentially a wrapper around the forward pass but there is not backpropagation

        Parameters
        ----------
        idx : torch.Tensor
            The batch of tokens [B x T] where B is the batch size and T is the number of tokens in a block
        max_new_tokens : int
            The maximum number of tokens to generate
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    def train_model(self, tokens, eval_iters=200, training_val_ratio=0.8, loss_report_interval=500):
        """
        Built-in unit test for training the model on a dataset reporting the training and validation loss

        Parameters
        ----------
        tokens : torch.Tensor
            The dataset of tokens
        eval_iters : int, optional
            The number of iterations to estimate the loss
        training_val_ratio : float, optional
            The ratio of the dataset to use for training (lower ratio means more data for validation)
        loss_report_interval : int, optional
            The interval to report the training and validation loss
        """
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
                losses = self._estimate_loss(eval_iters, training_tokens, validation_tokens)
                print(f"step {step}: train loss {losses[0]:.4f}, val loss {losses[1]:.4f}")

    @torch.no_grad()
    def _estimate_loss(self, eval_iters, training_data, validation_data):
        """
        Returns the loss of the model on a training and validation dataset

        Parameters
        ----------
        eval_iters : int
            The number of iterations to estimate the loss
        training_data : torch.Tensor
            The training dataset [B x T] where B is the batch size and T is the number of tokens in a block
        validation_data : torch.Tensor
            The validation dataset [B x T]
        """
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

# python lm_model.py -t tiktoken -m o200k_base -s models/threebody/200k_base -d data/threebody.txt 384 64 256 3e-4 5000 6 6 0.2
def sample(data, batch_size, block_size):
    starting_indices = torch.randint(len(data) - block_size, (batch_size,))
    sample = torch.stack([data[start_idx:start_idx+block_size] for start_idx in starting_indices])
    target = torch.stack([data[start_idx+1:start_idx+block_size+1] for start_idx in starting_indices])
    return sample, target

def useTiktoken(filename, model_name="o200k_base"):
    tokenizer = tiktoken.get_encoding(model_name)
    assert tokenizer.decode(tokenizer.encode("hello world")) == "hello world"
    with open(filename) as f:
        tokens = torch.tensor(tokenizer.encode(f.read()), dtype=torch.long, device=device)

    return tokenizer, tokens, tokenizer.n_vocab

def useLocal(filename, model_name="tokenizer_models/umb100k-1.model"):
    tokenizer = Tokenizer()
    tokenizer.load(model_name)
    assert tokenizer.decode(tokenizer.encode("hello world")) == "hello world"
    with open(filename) as f:
        tokens = torch.tensor(tokenizer.encode(f.read()), dtype=torch.long, device=device)

    return tokenizer, tokens, len(tokenizer._vocab)

    
if __name__ == "__main__":
    parser=argparse.ArgumentParser(
        description="""Train a language model on a dataset and generate text""")
    parser.add_argument('-t', '--tokenizer', type=str, default="tokenizer", help='Specify the tokenizer to use (default: tokenizer)')
    parser.add_argument('-m', '--tokenizer_model', type=str, default="tokenizer_models/umb100k-1.model", help='Specify the tokenizer model to use (default: tokenizer_models/umb100k-1.model)')
    parser.add_argument('-l', '--load_model', type=str, default="untrained", help='Specify the model to use [model_path] (default: untrained)')
    parser.add_argument('-s', '--save_model', type=str, default="default", help='Specify the model to save the model to [model_path] (default: same as load_model path, no_save: do not save model)')
    parser.add_argument('-d', '--data', type=str, default="data/threebody.txt", help='Specify the data to use for training (default: data/threebody.txt)')
    parser.add_argument('--no_train', type=bool, default=False, help='Do not train the model')
    parser.add_argument('params', nargs='*', default=[8, 4, 8, 1e-3, 5000, 4, 3, 0.1], help='Training parameters for the model [embedding_size, batch_size, block_size, learning_rate, steps, head_count, layer_count, dropout]\n(default: [4, 8, 8, 1e-3, 5000, 4, 3, 0.1])')
    # python 
    args=parser.parse_args()
    print(args)
    if args.tokenizer == "tokenizer":
        tokenizer, tokens, vocab_size = useLocal(args.data, args.tokenizer_model)
    elif args.tokenizer == "tiktoken":
        tokenizer, tokens, vocab_size = useTiktoken(args.data, args.tokenizer_model)
    else:
        print("Invalid tokenizer: must be either 'tokenizer' or 'tiktoken'")
        exit()

    lm = LanguageModel(
        vocab_size=vocab_size,
        embedding_size=int(args.params[0]),
        batch_size=int(args.params[1]),
        block_size=int(args.params[2]),
        learning_rate=float(args.params[3]),
        steps=int(args.params[4]),
        head_count=int(args.params[5]),
        layer_count=int(args.params[6]),
        dropout=float(args.params[7])
    )

    if args.load_model != "untrained":
        try:
            lm.load_state_dict(torch.load(args.load_model))
        except:
            print("Error: Model not found")
            exit()
    else:
        print("Warning: Using untrained model")
    
    if not args.no_train:
        lm.train_model(tokens)
    
    start_idx, _ = sample(tokens, lm.batch_size, lm.block_size)
    outputs = lm.generate(start_idx, max_new_tokens=400)[0].tolist()
    print(f"Prompt:\n{tokenizer.decode(start_idx[0].tolist())}\nGenerated Response:\n{tokenizer.decode(outputs)}")
    
    if args.save_model == "default":
        if args.load_model == "untrained":
            print("Warning: Model not saved")
        else:
            torch.save(lm.state_dict(), args.load_model)
    elif args.save_model == "no_save":
        print("Warning: Model not saved")
    else:
        torch.save(lm.state_dict(), args.save_model) 

    

    
       