import numpy as np
import torch

# Set the device to use for training
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("Warning: Using CPU for training; consider using a GPU for faster training")

def sample(data_path, batch_size, block_size):
    data = np.memmap(data_path, dtype=np.uint32, mode='r')
    '''attention_mask = data[0:block_size:3]
    input_ids = data[1:block_size+1:3]
    token_type_ids = data[2:block_size+2:3]
    print(attention_mask, input_ids, token_type_ids)'''

    starting_indices = torch.randint(len(data) // (block_size * 3) // 3, (batch_size,))
    print(len(data) // (block_size * 3))
    print(len(data))
    print(data[block_size*142*3+1:(block_size*142+block_size)*3+1:3])
    sample = torch.stack([torch.stack([torch.from_numpy((data[start_idx*block_size*3+offset:(start_idx*block_size+block_size)*3+offset:3]).astype(np.int64)) for offset in range(0, 3)]) for start_idx in starting_indices])
    target= torch.stack([torch.stack([torch.from_numpy((data[start_idx*block_size*3+offset:(start_idx*block_size+block_size)*3+offset:3]).astype(np.int64)) for offset in range(3, 6)]) for start_idx in starting_indices])
    sample = sample.pin_memory().to(device, non_blocking=True)
    target = target.pin_memory().to(device, non_blocking=True)
    return sample, target

if __name__ == "__main__":
    block_size = 128
    batch_size = 4
    s, t = sample("gpt/data/openwebtext/val.bin", batch_size, block_size)
    print(s[0, :2])