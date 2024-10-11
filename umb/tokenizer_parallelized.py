import pickle
from multiprocessing import shared_memory
import struct  # Used to pack/unpack the length
from multiprocessing import shared_memory, Process, Lock, Manager
from multiprocessing import cpu_count, current_process, Queue
import time
import numpy as np
from tqdm import tqdm

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

# Helper function for BPE (Byte-Pair Encoding); replace all occurences of a pair of utf-8 characters with a symbolic replacement and returns the result
def merge(text_bytes, pair, replacement_id):
    result = []
    index = 0
    # Linearly search through all the consecutive pairs in text_bytes and rpelace them with a symbolic replacement
    while index < len(text_bytes):
        if index < len(text_bytes) - 1 and text_bytes[index] == pair[0] and text_bytes[index+1] == pair[1]:
            result.append(replacement_id)
            index += 2
        else:
            result.append(text_bytes[index])
            index += 1

    return result

def bpe_process(shared_vocab, shared_merges, tokens, chunk_count, pid, epochs, output_queue):
    chunk_size = len(tokens) // chunk_count
    start_position = pid * chunk_size
    end_position = (pid + 1) * chunk_size if pid != chunk_count - 1 else len(tokens)
    chunk = tokens[start_position:end_position]

    
    for i in range(epochs):
        print(f"Process {pid} - Epoch {i}: {shared_vocab}")
        for (p0, p1), j in shared_merges.items():
            chunk = merge(chunk, (p0, p1), j)

        counts = get_stats(chunk)
        freq_pair = max(counts, key=lambda x: counts[(x[0], x[1])])
        idx = len(shared_vocab) + i
        chunk = merge(chunk, freq_pair, idx)
        shared_merges[freq_pair] = idx
        shared_vocab[idx] = shared_vocab[freq_pair[0]] + shared_vocab[freq_pair[1]]
        

    tokens[start_position:start_position+len(chunk)] = chunk
    output_queue.put(None)
    


VOCAB_SIZE = 800

if current_process().name == "MainProcess":
    with open("data/threebody.txt", "r") as f:
        text = f.read()

    # Tokenize the text into utf-8 bytes
    tokens = list(text.encode("utf-8"))

    manager = Manager()
    shared_vocab = manager.dict()
    shared_vocab = {idx: bytes([idx]) for idx in range(256)}
    shared_merges = manager.dict()
    token_array = np.array(tokens, dtype=np.int32)
    shared_tokens = shared_memory.SharedMemory(create=True, size=token_array.nbytes)
    np_array = np.ndarray(token_array.shape, dtype=token_array.dtype, buffer=shared_tokens.buf)
    np_array[:] = token_array[:]

    processes = []

    output_queue = Queue()

    num_processes = cpu_count()
    iteration_count = 100

    for pid in range(4):
        _process = Process(target=bpe_process, args=(shared_vocab, shared_merges, np_array, num_processes, pid, VOCAB_SIZE//num_processes, output_queue,))
        processes.append(_process)
        _process.start()

    for _process in processes:
        _process.join()

    print("Done")

    shared_tokens.close()
    shared_tokens.unlink()

    
