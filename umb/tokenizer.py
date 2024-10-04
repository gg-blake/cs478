"""
Name : tokenizer.py
Description : Encodes a string of text to a series of tokens where tokens are common groupings of characters and decodes a series of tokens into text. Creates a lookup table for tokens as well.
Author : Blake Moody
Date : 9-9-2024
"""
import regex as re
from tqdm import tqdm
import unicodedata
import pickle
from multiprocessing import shared_memory
import struct  # Used to pack/unpack the length
import multiprocessing
import time
import numpy as np
import os

import math
def clamp(value, min_value, max_value):
  return max(min(value, max_value), min_value)

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

class Tokenizer:
    def __init__(self, pattern=None):
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self._merges = {}
        self._special_tokens = {}
        self._vocab = self._build_vocab()
        self._mergeable_ranks = {}
    
    # Helper method to initialize an instance's vocabulary
    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self._merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self._special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
    
    def _build_mergeable_ranks(self, text):
        loader = tqdm(total=len(text), desc="Building mergeable ranks", unit="chars")
        text_chunks = re.findall(self.compiled_pattern, text)
        encoded_chunks = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        for chunk in encoded_chunks:
            for p0, p1 in zip(chunk, chunk[1:]):
                if (p0, p1) not in self._mergeable_ranks.keys():
                    self._mergeable_ranks[(p0, p1)] = 1
                    continue

                self._mergeable_ranks[(p0, p1)] = self._mergeable_ranks[(p0, p1)] + 1
                loader.update()

        loader.close()


    # Train the tokenizer on the given text
    # NOTE: This training method will also force prevent merges between different chunks of text. Chunks are formed by the regex pattern used by GPT-4
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256

        # Only add vocab_size number of symbols to the vocabulary
        num_merges = vocab_size - len(self._vocab)

        # Split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # Input text preprocessing
        # Convert the string to a list of utf-8 bytes
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # Store the resulting vocab table and merge table in temporary variables
        tmp_vocab = self._vocab.copy()
        tmp_merges = self._merges.copy()

        print(f"Training on {len(ids)} chunks of text...")
        loader = tqdm(total=vocab_size, desc="Training BPE", unit="merges")
        #loader.update(n=len(tmp_vocab))
        i = 0
        # BPE: Iteratively merge the most common consecutive pairings of bytes
        
        while len(tmp_vocab) < vocab_size:
            # Get the frequency stats for all chunks of text
            text_bytes_freq = {}
            for chunk_ids in ids:
                text_bytes_freq = self._pair_freq(chunk_ids, text_bytes_freq)

            # Store the most freq pair
            freq_pair = max(text_bytes_freq, key=lambda x: text_bytes_freq[(x[0], x[1])])
            idx = 256 + i
            
            ids = [self._merge(chunk_ids, freq_pair, idx) for chunk_ids in ids]
            tmp_merges[freq_pair] = idx
            tmp_vocab[idx] = tmp_vocab[freq_pair[0]] + tmp_vocab[freq_pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {freq_pair} -> {idx} ({tmp_vocab[idx]}) had {text_bytes_freq[freq_pair]} occurrences")

            

            loader.update()
            i += 1

        loader.close()

        # Save instance variables
        self._merges = tmp_merges # used in encode()
        self._vocab = tmp_vocab   # used in decode()



    # Prints the stats of a call to train()
    def train_stats(self, training_text, encode_text, vocab_size):
        # Reecord the old vocabulary and data sizes
        data = self.encode(encode_text)
        data_chars = sorted(list(set(data)))
        data_vocab_size_untrained = len(data_chars)
        data_size_untrained = len(data)

        # Run the trainer
        self.train(training_text, 300)

        # Record the new vocabulary and data sizes
        data = self.encode(encode_text)
        data_chars = sorted(list(set(data)))
        data_vocab_size_trained = len(data_chars)
        data_size_trained = len(data)

        # Print the stats
        print(f"Untrained Vocab Size: {data_vocab_size_untrained}\nUntrained Data Size: {data_size_untrained}\nTrained Vocab Size: {data_vocab_size_trained}\nTrained Data Size: {data_size_trained}\nRatio: {data_size_untrained/data_size_trained}")

    # Helper function for the encode() method: Returns a list of tokens (integers) encoded from the given raw bytes
    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            freq = self._pair_freq(ids)
            recent_pair = min(freq, key=lambda x: self._merges.get(x, float("inf")))
            if recent_pair not in self._merges:
                break

            idx = self._merges[recent_pair]
            ids = self._merge(ids, recent_pair, idx)

        return ids
    
    # Returns a list of tokens encoded, using its vocabulary, of a given text (string)
    def encode(self, text):
        # Split text into chunks of text with a given regex pattern (In this case its the GPT-4 pattern)
        text_chunks = re.findall(self.compiled_pattern, text)
        # Encode chunks of text separately, then rejoin and return the result
        ids = []
        for chunk in text_chunks:
            # Convert the chunk of text to raw bytes
            chunk_bytes = chunk.encode("utf-8")
            # Encode the chunk of raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)

        return ids

    # Returns a text representation (string) of the given sequence of tokens
    def decode(self, ids):
        tokens = b"".join(self._vocab[i] for i in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    # Helper function for BPE (Byte-Pair Encoding); replace all occurences of a pair of utf-8 characters with a symbolic replacement and returns the result
    def _merge(self, text_bytes, pair, replacement_id):
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
    
    # Helper function for BPE (Byte-Pair Encoding); replace all occurences of a pair of utf-8 characters with a symbolic replacement and returns the result
    def _merge_update(self, text_bytes, pair, replacement_id):
        result = []
        index = 1
        # Linearly search through all the consecutive pairs in text_bytes and rpelace them with a symbolic replacement
        self._mergeable_ranks[(pair[0], pair[1])] = 0
        pivot = 0 # The index of the last replacement
        replacements_found = 0
        while index < len(text_bytes):
            current_pair = (text_bytes[index-1], text_bytes[index]) if index > 0 else None
            current_pair_left = (text_bytes[index-2], text_bytes[index-1]) if index > 1 else None
            #current_pair_left = (result[-1], text_bytes[index-1]) if len(result) > 0 else None
            current_pair_right = (text_bytes[index], text_bytes[index+1]) if index < len(text_bytes) - 1 else None
            if current_pair == pair:
                replacements_found += 1
                if len(result) > 0:
                    # (left_char, *) -> +1
                    self._mergeable_ranks[(result[-1], replacement_id)] = self._mergeable_ranks[(result[-1], replacement_id)] + 1 if (result[-1], replacement_id) in self._mergeable_ranks.keys() else 1

                # (left neighbor char, pair_left_char) -> -1
                if current_pair_left is not None:
                    self._mergeable_ranks[current_pair_left] = int(clamp(self._mergeable_ranks[current_pair_left] - 1, 0, math.inf))

                # (right neighbor char, pair_right_char) -> -1
                if current_pair_right is not None:
                    self._mergeable_ranks[current_pair_right] = int(clamp(self._mergeable_ranks[current_pair_right] - 1, 0, math.inf))
                    
                
                result.append(replacement_id)
                pivot = len(result) - 1
                index += 2
            else:
                result.append(text_bytes[index-1])
                # (*, right_char) -> +1
                if len(result) - 1 > pivot and result[-2] == replacement_id:
                    self._mergeable_ranks[(replacement_id, result[-1])] = self._mergeable_ranks[(replacement_id, result[-1])] + 1 if (replacement_id, result[-1]) in self._mergeable_ranks.keys() else 1

                index += 1

        

        '''index = 0
        while index < len(result):
            if index < len(result) - 1 and result[index+1] == replacement_id:
                if (result[index], replacement_id) not in self._mergeable_ranks.keys():
                    self._mergeable_ranks[(result[index], replacement_id)] = 1
                    index += 2
                    continue

                self._mergeable_ranks[(result[index], result[index+1])] = self._mergeable_ranks[(result[index], result[index+1])] + 1
                index += 2
            else:
                index += 1'''
                
        return result

    # Helper function for BPE (Byte-Pair Encoding); Returns an dictionary containing all existing consecutive character pairs as keys and their respective frequency as the value
    def _pair_freq(self, text_bytes, pair_freq_counts=None):
        pair_freq = {} if pair_freq_counts is None else pair_freq_counts
        for p0, p1 in zip(text_bytes, text_bytes[1:]):
            if (p0, p1) not in pair_freq.keys():
                pair_freq[(p0, p1)] = 1
                continue

            pair_freq[(p0, p1)] = pair_freq[(p0, p1)] + 1

        return pair_freq
    
    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("bpetokenizer v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self._special_tokens)}\n")
            for special, idx in self._special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self._merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self._merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self._vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self._vocab[idx0])
                    s1 = render_token(self._vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "bpetokenizer v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self._merges = merges
        self._special_tokens = special_tokens
        self._vocab = self._build_vocab()


if __name__ == "__main__":
    VOCAB_SIZE = 1000
    tokenizer = Tokenizer()
    tokenizer_training_data = ""

    
    files = [file for file in os.listdir("data")]
    loader = tqdm(total=len(files), desc="Compiling training data")
    for file in files:
        with open(f"data/{file}") as g:
            tokenizer_training_data += g.read()
        g.close()
        loader.update()
    loader.close()

    tokenizer._build_mergeable_ranks(tokenizer_training_data)
    tokenizer.train(tokenizer_training_data, VOCAB_SIZE)
    e = tokenizer.encode("Hello, my name is Blake. What is your name?")
    print(tokenizer._vocab)
    pair = max(tokenizer._mergeable_ranks, key=lambda x: tokenizer._mergeable_ranks[(x[0], x[1])])
    print(tokenizer._vocab[pair[0]], tokenizer._vocab[pair[1]])


    '''try:
        tokenizer.load("tokenizer_models/umb2M.model")
    except FileNotFoundError:
        try:
            tokenizer.train(tokenizer_training_data, VOCAB_SIZE)
        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
        tokenizer.save("tokenizer_models/umb2M")'''

# This is the command to download the umb website
# wget -m -k -K -E -l 7 -t 6 -w 5 https://www.umb.edu/