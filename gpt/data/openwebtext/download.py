import datasets
import tiktoken
import numpy as np
import os
from tqdm import tqdm

num_processes = 4

tokenizer = tiktoken.get_encoding('o200k_base')

if __name__ == "__main__":
    dataset = datasets.load_dataset('openwebtext', trust_remote_code=True, num_proc=num_processes)
    split_dataset = dataset['train'].train_test_split(test_size=0.0005, seed=42)
    split_dataset['validation'] = split_dataset.pop('test')

    def tokenize(example):
        ids = tokenizer.encode(example['text'])
        ids.append(tokenizer.eot_token)
        
        return {
            'ids': ids,
            'len': len(ids)
        }
    
    tokenized_dataset = split_dataset.map(
        tokenize,
        remove_columns='text',
        num_proc=num_processes
    )

    
    for split, dset in tokenized_dataset.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint32 # Change this based on the upper bound of tokenizer.n_vocab as a power of 2 (i.e. 200,000<=)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

