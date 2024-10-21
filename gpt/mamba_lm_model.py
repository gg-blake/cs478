import torch
import torch.nn as nn
from torch.optim import adamw
import tiktoken
import torch.nn.functional as F

class MambaRNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size, block_size, n_layers, dropout=0.5):
        super(MambaRNNLanguageModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.block_size = block_size

        # Embedding layer to convert token IDs to vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, device=device)

        # Positional encoding
        self.positional_encoding = nn.Embedding(block_size, embedding_dim, device=device)
        
        # RNN layer (You can switch to nn.LSTM for LSTM-based model)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout, device=device)
        
        # Fully connected layer to map the hidden state to output (vocab size)
        self.fc = nn.Linear(hidden_dim, vocab_size, device=device)

        self.layer_norm = nn.LayerNorm(hidden_dim, device=device)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden):
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)

        # Add positional encoding
        pos = torch.arange(0, x.size(1)).unsqueeze(0).repeat(x.size(0), 1).to(device)
        pos_embedded = self.dropout(self.positional_encoding(pos))
        embedded = embedded + pos_embedded
        
        # Pass through the RNN layer
        output, hidden = self.rnn(embedded, hidden)  # output: (batch_size, seq_length, hidden_dim)
        
        # Apply layer normalization
        output = self.layer_norm(output)

        # Apply fully connected layer to each output token
        output = self.fc(output)  # (batch_size, seq_length, vocab_size)
        
        return output, hidden

    def init_hidden(self):
        # Initialize the hidden state for the RNN (GRU)
        return torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(device)
    
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
        model.eval()
        hidden = model.init_hidden().to(device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, hidden = self(idx_cond, hidden)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Define hyperparameters
tokenizer = tiktoken.get_encoding("o200k_base")
vocab_size = tokenizer.n_vocab  # Vocabulary size (adjust this according to your dataset)
embedding_dim = 384  # Embedding dimensions
hidden_dim = 384     # Hidden dimensions for the RNN
n_layers = 2         # Number of layers in the RNN

# Training settings
learning_rate = 0.0001
num_epochs = 5000
batch_size = 8
block_size = 30  # Number of tokens in each training sequence
weight_decay = 0.01

# Device configuration (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model
model = MambaRNNLanguageModel(
    vocab_size=vocab_size, 
    embedding_dim=embedding_dim, 
    hidden_dim=hidden_dim,
    batch_size=batch_size,
    block_size=block_size,
    n_layers=n_layers,
    dropout=0.2
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = adamw.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def train(model, tokens, vocab_size, num_epochs):
    model.train()
    
    for epoch in range(num_epochs):
        hidden = model.init_hidden()
        
        input_seq, target_seq = sample(tokens, batch_size, block_size)
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        
        # Forward pass
        output, hidden = model(input_seq, hidden)
        loss = criterion(output.view(-1, vocab_size), target_seq.view(-1))
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            


def sample(data, batch_size, block_size):
    starting_indices = torch.randint(len(data) - block_size, (batch_size,))
    sample = torch.stack([data[start_idx:start_idx+block_size] for start_idx in starting_indices])
    target = torch.stack([data[start_idx+1:start_idx+block_size+1] for start_idx in starting_indices])
    return sample, target

if __name__ == "__main__":
    with open("data/threebody.txt", 'r') as f:
        text = f.read()
    
    tokens = torch.tensor(tokenizer.encode(text))
    # Assume data_loader is a PyTorch DataLoader with tokenized input data
    train(model, tokens, vocab_size, num_epochs)

    start_idx, _ = sample(tokens, batch_size, block_size)
    start_idx = start_idx.to(device)

    outputs = model.generate(start_idx, max_new_tokens=400)[0].tolist()
    print(f"Prompt:\n{tokenizer.decode(start_idx[0].tolist())}\nGenerated Response:\n{tokenizer.decode(outputs)}")

# Assume data_loader is a PyTorch DataLoader with tokenized input data
# train(model, data_loader, vocab_size, num_epochs)
