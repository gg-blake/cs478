import tiktoken
from tqdm import tqdm
import os
import datasets
import numpy as np
from transformers import BertTokenizer
from nltk.tokenize import sent_tokenize
import transformers
import random
import math
import torch
import os


tokenizer = tiktoken.get_encoding("o200k_base")
tokenizer._special_tokens["[SEP]"] = tokenizer.max_token_value + 1
block_size = 128
batch_size = 4
stride_size = 32
max_strides = math.ceil(block_size / stride_size) # This is how many unint32 will store an attention mask or token_type_ids 

output_size = {
    'attention_mask': max_strides,
    'input_ids': block_size,
    'token_type_ids': max_strides,
    'label': 1
}

#dataset = datasets.load_dataset("stas/openwebtext-10k", num_proc=8)
dataset = datasets.load_dataset("openwebtext", num_proc=8)

# owt by default only contains the 'train' split, so create a test split
global split_dataset
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test') # rename the test split to val


# For more details - https://huggingface.co/bert-base-uncased
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=block_size)

transformers.logging.set_verbosity_error()

def process(example):
    sentences = sent_tokenize(example["text"])
    return {'result': list(zip(sentences[0:-2], sentences[1:-1]))}

def process_rand(example):
    global split_dataset
    result = []

    for p0, p1 in example['result']:
        is_true = bool(np.random.binomial(n=1, p=0.5))
        if is_true:
            pair = tokenizer_bert(p0, p1, truncation="longest_first", padding="max_length")
        else:
            set_r = random.choice(['train', 'val'])
            text_r = random.choice(split_dataset[set_r])
            try:
                result_r = random.choice(text_r['result'])
                pair = tokenizer_bert(p0, result_r[1], truncation="longest_first", padding="max_length")
            except IndexError:
                continue

        result.append({
            'attention_mask': pair['attention_mask'],
            'input_ids': pair['input_ids'],
            'token_type_ids': pair['token_type_ids'],
            'label': is_true
        })

    return {'items': result}

# Flatten the sequences in each dictionary within the 'items' column
def explode_sequences(batch):
    # Prepare lists to store the column values
    attention_mask_list = []
    input_ids_list = []
    token_type_ids_list = []
    labels = []
    
    for row_items in batch["items"]:  # Each row_items is a list of dictionaries
        for item in row_items:  # Each item is a dictionary
            # Compress the 32 bits of the attention mask and sentence ids into list of unsigned 32-bit integers 
            att_chunks = [item["attention_mask"][i:i+stride_size] for i in range(0, max_strides * stride_size, stride_size)]
            ids_chunks = [item["token_type_ids"][i:i+stride_size] for i in range(0, max_strides * stride_size, stride_size)]
            uint_list_1 = [int("".join(map(str, chunk)), 2) for chunk in att_chunks]
            uint_list_2 = [int("".join(map(str, chunk)), 2) for chunk in ids_chunks]
            # Process each sequence in the item
            attention_mask_list.append(uint_list_1)
            input_ids_list.append(item["input_ids"])
            token_type_ids_list.append(uint_list_2)
            labels.append([int(item['label'])])
                
    
    # Return a dictionary with each column as a list
    return {
        "attention_mask": attention_mask_list,
        "input_ids": input_ids_list,
        "token_type_ids": token_type_ids_list,
        "label": labels
    }

def output_transform():
    global split_dataset
    split_dataset = split_dataset.map(process, remove_columns=['text'], num_proc=5)
    split_dataset = split_dataset.map(process_rand, remove_columns=['result'], num_proc=5)
    split_dataset = split_dataset.map(explode_sequences, remove_columns=['items'], batched=True, num_proc=5)

def save_to_disk():
    global split_dataset
    output_transform()
    for split, dset in split_dataset.items():
        for data_type in dset.column_names:
            idx = 0
            arr_len = dset.num_rows * output_size[data_type]
            filename = os.path.join(os.path.dirname(__file__), f'{split}-{data_type}.bin')
            dtype = np.uint32 # Change this based on the upper bound of tokenizer.n_vocab as a power of 2 (i.e. 200,000<=)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            total_batches = 1024

            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch[data_type])
                if data_type == 'label':
                    print(len(arr_batch))
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

def tensor_to_binary(x, bit_num):
    return x.unsqueeze(-1).bitwise_and(
            2**torch.arange(bit_num-1,-1,-1).to(x.device, x.dtype)
        ).ne(0).byte()

def uncompress(x):
    if len(x) < block_size:
        return torch.flatten(
            tensor_to_binary(x, stride_size)
            )
    return x

def slice_from_memory(x: np.memmap, scale: int, offset: int) -> torch.Tensor:
    return torch.from_numpy(
        np.ndarray(
            shape=(scale,), 
            buffer=x[scale*offset:scale*offset+scale],
            dtype=np.uint32
        ).astype(np.int32)
    )

def sample(data_dir, split, batch_size, block_size):
    data = [np.memmap(os.path.join(data_dir, f"{split}-{filename}.bin"), dtype=np.uint32, mode='r') for filename in ['attention_mask', 'input_ids', 'token_type_ids', 'label']]
    data_scales = [block_size // (len(data[1]) // len(data[i])) for i in range(len(data))]
    indices = torch.randint(len(data[3]) - 1, (batch_size,))
    sample = torch.stack([torch.stack([uncompress(slice_from_memory(data[i], data_scales[i], int(start_idx.item()))) for i in range(len(data_scales) - 1)]) for start_idx in indices])
    target = torch.tensor([data[-1][int(start_idx.item())] for start_idx in indices])
    return sample, target

if __name__ == "__main__":
    save_to_disk()