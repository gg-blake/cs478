import tiktoken
from model import LanguageModel
from model_config import *
import argparse
import torch
import torch.nn.functional as F
import sys
import datasets
import os
from train import sample
from time import time
from tqdm import tqdm

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

def load_tokenizer(model_name):
    print(model_name)
    tokenizer = tiktoken.get_encoding(model_name)
    assert tokenizer.decode(tokenizer.encode("hello world")) == "hello world"
    return tokenizer, tokenizer.n_vocab

def generate(model, idx, max_new_tokens):
    """
    Generate text from the model given an initial set of sample tokens; it's essentially a wrapper around the forward pass but there is not backpropagation

    Parameters
    ----------
    idx : torch.Tensor
        The batch of tokens [B x T] where B is the batch size and T is the number of tokens in a block
    max_new_tokens : int
        The maximum number of tokens to generate
    """
    for _ in tqdm(range(max_new_tokens), desc="Generating tokens"):
        idx_cond = idx[:, -model.block_size:]
        logits, loss = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


if __name__ == "__main__":
    parser=argparse.ArgumentParser(
        description="""Train a language model on a dataset and generate text""")
    parser.add_argument('-l', '--load_model', type=str, default="untrained", help='Specify the model to use [model_path] (default: untrained)')
    parser.add_argument('-d', '--data_dir', type=str, default=TRAIN_DATA_PATH, help=f'Specify the data to use for training (default: {TRAIN_DATA_PATH})')
    parser.add_argument('params', nargs='*', default=LM_MODEL_CONFIG, help=f'Training parameters for the model [embedding_size, batch_size, block_size, learning_rate, steps, head_count, layer_count, dropout]\n(default: {LM_MODEL_CONFIG})')
    args=parser.parse_args()

    tokenizer = tiktoken.get_encoding(TOKENIZER_MODEL)

    lm = LanguageModel(
        vocab_size=tokenizer.n_vocab,
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
            checkpoint = torch.load(args.load_model)
            lm.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint["epoch"]
        except:
            print("Error: Model not found")
            exit()
    else:
        print("Error: Using untrained model. Exiting.")
        exit()

    idx, _ = sample(os.path.join(os.path.dirname(__file__), args.data_dir, "validation.bin"), lm.batch_size, lm.block_size)
    max_tokens = input("Max number of generated tokens: ")
    os.system('clear')
    start = time()
    output = generate(lm, idx, max_new_tokens=int(max_tokens))
    output = tokenizer.decode(output[0].tolist())
    end = time()
    with open("output.txt", "w") as f:
        f.write(output)
        f.close()
    
    print(f"Output saved successfully to output.txt\nGenerated in {(end-start)/1000} seconds")

    