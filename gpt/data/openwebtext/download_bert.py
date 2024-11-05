import tiktoken
from tqdm import tqdm
import os
import datasets
import numpy as np
from transformers import BertTokenizer
from nltk.tokenize import sent_tokenize
import transformers
import random


tokenizer = tiktoken.get_encoding("o200k_base")
tokenizer._special_tokens["[SEP]"] = tokenizer.max_token_value + 1
data_path = "gpt/data/openwebtext/train.bin"
data = np.memmap(data_path, dtype=np.uint32, mode='r')
block_size = 128
batch_size = 4


dataset = datasets.load_dataset("stas/openwebtext-10k", num_proc=8)

# owt by default only contains the 'train' split, so create a test split
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test') # rename the test split to val


# For more details - https://huggingface.co/bert-base-uncased
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=block_size)

transformers.logging.set_verbosity_error()
def process(example):
    sentences = sent_tokenize(example["text"])
    return {'result': list(zip(sentences[0:-2], sentences[1:-1]))}

def process_rand(example):
    
    result = []

    for p0, p1 in example['result']:
        if bool(np.random.binomial(n=1, p=0.5)):
            pair = tokenizer_bert(p0, p1, truncation="longest_first", padding="max_length")
        else:
            set_r = random.choice(['train', 'val'])
            text_r = random.choice(output[set_r])
            try:
                result_r = random.choice(text_r['result'])
                pair = tokenizer_bert(p0, result_r[1], truncation="longest_first", padding="max_length")
            except IndexError:
                continue

        '''bits_att_mask = sum(bit << i for i, bit in enumerate(reversed(pair['attention_mask'])))
        bits_token_type = sum(bit << i for i, bit in enumerate(reversed(pair['attention_mask'])))'''

        result.append({
            'attention_mask': pair['attention_mask'],
            'input_ids': pair['input_ids'],
            'token_type_ids': pair['token_type_ids']
        })

    return {'items': result}

def test(row):
    result = []
    '''for item in row['items']:
        print(item, "\n")'''
    
    return [({"attention_mask": dict(item)["attention_mask"], "input_ids": dict(item)["input_ids"], "token_type_ids": dict(item)["token_type_ids"]} for item in items) for items in row["items"]]

output = split_dataset.map(process, remove_columns=['text'], num_proc=8)
output = output.map(process_rand, remove_columns=['result'], num_proc=8)

# Flatten the sequences in each dictionary within the 'items' column
def explode_sequences(batch):
    # Prepare lists to store the column values
    attention_mask_list = []
    input_ids_list = []
    token_type_ids_list = []
    
    for row_items in batch["items"]:  # Each row_items is a list of dictionaries
        for item in row_items:  # Each item is a dictionary
            # Process each sequence in the item
            attention_mask_list.append(item["attention_mask"])
            input_ids_list.append(item["input_ids"])
            token_type_ids_list.append(item["token_type_ids"])
                
    
    # Return a dictionary with each column as a list
    return {
        "attention_mask": attention_mask_list,
        "input_ids": input_ids_list,
        "token_type_ids": token_type_ids_list,
    }

# Use map with batched=True to apply the function to each row
output = output.map(
    explode_sequences,
    remove_columns=['items'],
    batched=True,
    num_proc=5
)

# Flatten the nested lists in the resulting dataset
print(output["train"].num_rows)

for split, dset in output.items():
    arr_len = np.sum(block_size * 3 * dset.num_rows, dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint32 # Change this based on the upper bound of tokenizer.n_vocab as a power of 2 (i.e. 200,000<=)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch_att, arr_batch_tokens, arr_batch_ids = np.concatenate(batch['attention_mask']), np.concatenate(batch['input_ids']), np.concatenate(batch['token_type_ids'])
        arr[idx : idx + len(arr_batch_att)] = arr_batch_att[:]
        arr[idx + len(arr_batch_att) : idx + len(arr_batch_tokens)] = arr_batch_tokens[:]
        arr[idx + len(arr_batch_tokens) : idx + len(arr_batch_ids)] = arr_batch_ids[:]
        idx += len(arr_batch_att) + len(arr_batch_tokens) + len(arr_batch_ids)
    arr.flush()