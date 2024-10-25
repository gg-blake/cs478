from tokenizer import Tokenizer
import torch
from torch.optim import adamw
import tiktoken
from math import floor
from tqdm import tqdm
import argparse
from model_config import *
import os
from model import LanguageModel
import numpy as np

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

epoch = 0

# Set the device to use for training
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("Warning: Using CPU for training; consider using a GPU for faster training")

def sample(data_path, batch_size, block_size):
    data = np.memmap(data_path, dtype=np.uint32, mode='r')
    starting_indices = torch.randint(len(data) - block_size, (batch_size,))
    sample = torch.stack([torch.from_numpy((data[start_idx:start_idx+block_size]).astype(np.int64)) for start_idx in starting_indices])
    target = torch.stack([torch.from_numpy((data[start_idx+1:start_idx+block_size+1]).astype(np.int64)) for start_idx in starting_indices])
    sample, target = sample.pin_memory().to(device, non_blocking=True), target.pin_memory().to(device, non_blocking=True)
    return sample, target

def train(model, training_data_path, validation_data_path, eval_iters=200, training_val_ratio=0.8, loss_report_interval=500):
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

    global epoch
    optimizer = adamw.AdamW(model.parameters(), lr=model.learning_rate)
    loader = tqdm(total=model.steps)
    loader.update(n=epoch)
    for step in range(epoch, model.steps):
        try:
            optimizer.zero_grad()
            s, t = sample(training_data_path, model.batch_size, model.block_size)
            logits, loss = lm(s, t)
            loss.backward()
            optimizer.step()
            if step % loss_report_interval == 0:
                losses = _estimate_loss(lm, eval_iters, training_data_path, validation_data_path)
                loader.set_description(f"Step {step}: train loss {losses[0]:.4f}, val loss {losses[1]:.4f}")

            loader.update()
        except KeyboardInterrupt or ValueError:
            epoch = step
            break

    loader.close()

@torch.no_grad()
def _estimate_loss(model, eval_iters, training_data_path, validation_data_path):
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
    for i, split in enumerate([training_data_path, validation_data_path]):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = sample(split, model.batch_size, model.block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[i] = losses.mean()
    # Enable dropout and layer normalization after model validation
    model.train()
    return out

if __name__ == "__main__":
    parser=argparse.ArgumentParser(
        description="""Train a language model on a dataset and generate text""")
    parser.add_argument('-l', '--load_model', type=str, default="untrained", help='Specify the model to use [model_path] (default: untrained)')
    parser.add_argument('-s', '--save_model', type=str, default="default", help='Specify the model to save the model to [model_path] (default: same as load_model path, no_save: do not save model)')
    parser.add_argument('-d', '--data_dir', type=str, default=TRAIN_DATA_PATH, help=f'Specify the data to use for training (default: {TRAIN_DATA_PATH})')
    parser.add_argument('--no_train', type=bool, default=False, help='Do not train the model')
    parser.add_argument('params', nargs='*', default=LM_MODEL_CONFIG, help=f'Training parameters for the model [embedding_size, batch_size, block_size, learning_rate, steps, head_count, layer_count, dropout]\n(default: {LM_MODEL_CONFIG})')
    # python 
    args=parser.parse_args()

    tokenizer = tiktoken.get_encoding(TOKENIZER_MODEL)

    if not os.path.exists(args.data_dir):
        print("Error: Data path does not exist. Exiting.")
        exit()

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

    total_params = sum(p.numel() for p in lm.parameters())
    print(f"Number of parameters: {total_params}")
    
    if args.load_model != "untrained":
        try:
            checkpoint = torch.load(args.load_model)
            lm.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint["epoch"]
        except:
            print("Error: Model not found")
            exit()
    else:
        print("Warning: Using untrained model")
    
    if not args.no_train:
        training_data_path = os.path.join(os.path.dirname(__file__), args.data_dir, "train.bin")
        validation_data_path = os.path.join(os.path.dirname(__file__), args.data_dir, "validation.bin")
        train(lm, training_data_path, validation_data_path)
    
    if args.save_model == "default":
        if args.load_model == "untrained":
            print("Warning: Model not saved")
        else:
            torch.save({
                "epoch": epoch,
                "model_state_dict": lm.state_dict()
            }, args.load_model)
    elif args.save_model == "no_save":
        print("Warning: Model not saved")
    else:
        torch.save({
            "epoch": epoch,
            "model_state_dict": lm.state_dict()
        }, args.save_model)
    
    