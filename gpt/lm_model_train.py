from tokenizer import Tokenizer
import torch
from torch.optim import adamw
import tiktoken
from math import floor
from tqdm import tqdm
import argparse
from lm_config import *
import os
from lm_model import LanguageModel

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

def sample(data, batch_size, block_size):
    starting_indices = torch.randint(len(data) - block_size, (batch_size,))
    sample = torch.stack([data[start_idx:start_idx+block_size] for start_idx in starting_indices])
    target = torch.stack([data[start_idx+1:start_idx+block_size+1] for start_idx in starting_indices])
    return sample, target

def train(model, tokens, eval_iters=200, training_val_ratio=0.8, loss_report_interval=500):
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

    optimizer = adamw.AdamW(model.parameters(), lr=model.learning_rate)
    loader = tqdm(total=model.steps)
    for step in range(model.steps):
        optimizer.zero_grad()
        s, t = sample(training_tokens, 4, 8)
        logits, loss = lm(s, t)
        loss.backward()
        optimizer.step()
        if step % loss_report_interval == 0:
            losses = model._estimate_loss(eval_iters, training_tokens, validation_tokens)
            loader.set_description(f"Step {step}: train loss {losses[0]:.4f}, val loss {losses[1]:.4f}")

        loader.update()

    loader.close()

@torch.no_grad()
def _estimate_loss(model, eval_iters, training_data, validation_data):
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
    model.eval()
    for i, split in enumerate([training_data, validation_data]):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = sample(split, model.batch_size, model.block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[i] = losses.mean()
    # Enable dropout and layer normalization after model validation
    model.train()
    return out

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
    parser.add_argument('-t', '--tokenizer', type=str, default=TOKENIZER_NAME, help=f'Specify the tokenizer to use (default: {TOKENIZER_NAME})')
    parser.add_argument('-m', '--tokenizer_model', type=str, default=TOKENIZER_MODEL, help=f'Specify the tokenizer model to use (default: {TOKENIZER_MODEL})')
    parser.add_argument('-l', '--load_model', type=str, default="untrained", help='Specify the model to use [model_path] (default: untrained)')
    parser.add_argument('-s', '--save_model', type=str, default="default", help='Specify the model to save the model to [model_path] (default: same as load_model path, no_save: do not save model)')
    parser.add_argument('-d', '--data', type=str, default=TRAIN_DATA_PATH, help=f'Specify the data to use for training (default: {TRAIN_DATA_PATH})')
    parser.add_argument('--no_train', type=bool, default=False, help='Do not train the model')
    parser.add_argument('params', nargs='*', default=LM_MODEL_CONFIG, help=f'Training parameters for the model [embedding_size, batch_size, block_size, learning_rate, steps, head_count, layer_count, dropout]\n(default: {LM_MODEL_CONFIG})')
    # python 
    args=parser.parse_args()
    print(args)

    

    if not os.path.exists(args.data):
        print("Error: Data path does not exist. Exiting.")
        exit()


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
    
    