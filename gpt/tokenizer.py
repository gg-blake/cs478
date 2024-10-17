"""
Name : tokenizer.py
Description : Encodes a string of text to a series of tokens where tokens are common groupings of characters and decodes a series of tokens into text. Creates a lookup table for tokens as well.
Author : Blake Moody
Date : 9-9-2024
"""
import multiprocessing
import regex as re
from tqdm import tqdm
import unicodedata
import pickle
import multiprocessing
import struct  # Used to pack/unpack the length
import time
import numpy as np
import os
from benchmark import benchmark
import json

import math
def clamp(value, min_value, max_value):
  return max(min(value, max_value), min_value)

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# Helper function for BPE (Byte-Pair Encoding); Returns an dictionary containing all existing consecutive character pairs as keys and their respective frequency as the value
def _pair_freq(text_bytes, pair_freq_counts=None):
    pair_freq = {} if pair_freq_counts is None else pair_freq_counts
    for p0, p1 in zip(text_bytes, text_bytes[1:]):
        if (p0, p1) not in pair_freq.keys():
            pair_freq[(p0, p1)] = 1
            continue

        pair_freq[(p0, p1)] = pair_freq[(p0, p1)] + 1

    return pair_freq

def _increase_frequency(freq, pair):
    if pair not in freq.keys():
        freq[pair] = 1
    else:
        freq[pair] += 1

def _decrease_frequency(freq, pair):
    if pair not in freq.keys():
        freq[pair] = 0
    else:
        freq[pair] -= 1

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

def _merge_naive(text_bytes, pair, replacement_id):
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

def _merge_naive_worker(shared_ids, text_bytes, pair, replacement_id, start, end, shared_diff):
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
                
        diff = len(text_bytes) - len(result)
        print(start-shared_diff.value, end-shared_diff.value-diff)
        shared_ids[start-shared_diff.value:end-shared_diff.value-diff] = result
        shared_diff.value -= diff
        return result

def _merge(freq, text_bytes, pair, replacement_id):
    result = []
    freq[(replacement_id, replacement_id)] = 0
    count = 0
    index = 0
    while index < len(text_bytes):
        if index < len(text_bytes) - 1 and text_bytes[index] == pair[0] and text_bytes[index+1] == pair[1]:
            count += 1
            result.append(replacement_id)
            
            if index - 2 >= 0:
                LNPO = (text_bytes[index-1], text_bytes[index]) # Left Neighbor Pair Old
                LRP = (text_bytes[index-1], replacement_id)
                _decrease_frequency(freq, LNPO)
                # Check if there was a matching pair immediately before
                if text_bytes[index-2] == pair[0] and text_bytes[index-1] == pair[1]:
                    # If there isn't an immediate neighboring pair,
                    _decrease_frequency(freq, LNPO) # This will prevent a redundant increase of the pair (text_bytes[index-1], text_bytes[index])
                    _decrease_frequency(freq, (replacement_id, text_bytes[index])) # We need to undo a frequency update of the last pair (replacement_id, text_bytes[index])
                else:
                    # If there isn't an immediate neighboring pair, then we can increment the replacement left pair by one
                    _increase_frequency(freq, LRP)
            elif index - 1 >= 0:
                # Check if there was a matching pair immediately before
                LNPO = (text_bytes[index-1], text_bytes[index]) # Left Neighbor Pair Old
                _decrease_frequency(freq, LNPO)
                LRP = (text_bytes[index-1], replacement_id)
                _increase_frequency(freq, LRP)

            if index + 3 < len(text_bytes):
                RNPO = (text_bytes[index+1], text_bytes[index+2]) # Right Neighbor Pair Old
                RRP = (replacement_id, text_bytes[index+2]) # Right Replacement Pair
                _decrease_frequency(freq, RNPO)
                if  not (text_bytes[index+2] == pair[0] and text_bytes[index+3] == pair[1]):
                    _increase_frequency(freq, RRP)
                else:
                    _increase_frequency(freq, (replacement_id, replacement_id))
            elif index + 2 < len(text_bytes):
                RNPO = (text_bytes[index+1], text_bytes[index+2]) # Right Neighbor Pair Old
                RRP = (replacement_id, text_bytes[index+2]) # Right Replacement Pair
                _decrease_frequency(freq, RNPO)
                _increase_frequency(freq, RRP)

            index += 2

        else:
            result.append(text_bytes[index])
            index += 1

    

    freq[pair] = 0

    return result

def merge_chunk(freq, text_bytes, pairs, replacement_id):
    manager = multiprocessing.Manager()
    shared_ids = multiprocessing.Array('i', text_bytes, lock=False)
    shared_freq = manager.dict(freq)
    
    processes = []
    for i, pair in enumerate(pairs):
        # Create a new process for each pair to replace
        p = multiprocessing.Process(target=_merge, args=(shared_freq, shared_ids, pair, replacement_id+i))
        processes.append(p)
        p.start()
        
    # Ensure all processes complete
    for p in processes:
        p.join()
    
    return shared_ids[:]

class Tokenizer:
    """
    A class used to convert text to numbers (tokens)

    Attributes
    ----------
    pattern : str
        a regular expression to chunk text input into more logical segments
    compiled_pattern : Pattern[str]
        a compiled regular expression pattern from the given regular expression string
    _merges : dict[tuple[int, int], int]
        a mapping of token pairs to their symbolic token representation
    _special_tokens : dict[int, str]
        a mapping of tokens to special text characters (end of line, new line, etc.)
    _vocab : dict[int, str]
        a mapping of tokens to utf-8 text
    _mergeable_ranks : dict[tuple[int, int], int]
        a mapping of token pairs to their current frequency in the merged training data

    Methods
    -------
    _build_vocab()
        Returns the initial mapping of tokens to utf-8 text after starting with 256-character
        and applying the initial merges and adding special tokens to the vocabulary
    train_naive(text, vocab_size, verbose=False)
        Updates the tokenizer vocabulary based on the given text;
        performs Byte-Pair Encoding (BPE) on the text and iteratively merges the most common pairings 
        of tokens. Calculates the most frequent pair every iteration in linear time.
    train(text, vocab_size, verbose=False)
        Updates the tokenizer vocabulary based on the given text;
        performs Byte-Pair Encoding (BPE) on the text and iteratively merges the most common pairings 
        of tokens. Calculates the most frequent pair every iteration in constant time.
    _encode_chunk(text_bytes)
        Helper function for the encode() method: Returns a list of tokens (integers) encoded from the given raw bytes
    encode(text)
        Returns a list of tokens given an input string
    decode(ids)
        Returns a string representation of a given list of tokens
    save(file_prefix)
        Saves the instance's vocabulary and merge history to a text file ending in .model with the given file prefix
    load(model_file)
        Loads the vocabulary and merge history from a text file given a string; 
        model_file must include file suffix .model
    """
  
    def __init__(self, pattern=None):
        """
        Parameters
        ----------
        pattern : str
            The regular expression for chunking the model training data (default is None)
        """
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self._merges = {}
        self._special_tokens = {}
        self._vocab = self._build_vocab()
        self._mergeable_ranks = {}
    
    def _build_vocab(self):
        """Returns the initial mapping of tokens to utf-8 text after starting with 256-character
        and applying the initial merges and adding special tokens to the vocabulary
        """
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self._merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self._special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train_naive(self, text, vocab_size, verbose=False):
        """Updates the tokenizer vocabulary based on the given text;
        performs Byte-Pair Encoding (BPE) on the text and iteratively merges the most common pairings 
        of tokens. Calculates the most frequent pair every iteration in linear time.

        NOTE: This training method will also force prevent merges between different chunks of text. Chunks are formed by the regex pattern used by GPT-4

        Time Complexity: O(3mn) where m is the length of the text and n is the number of compression iterations

        Parameters
        ----------
        text : str
            The text to train the tokenizer
        vocab_size : int
            The number of iterations to perform a merging of the most frequent pair
        verbose : bool, optional
            When enabled, will print the result of every merging of a common pair, including its frequency and its new token replacement
        """
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
                text_bytes_freq = _pair_freq(chunk_ids, text_bytes_freq)

            # Store the most freq pair
            freq_pair = max(text_bytes_freq, key=lambda x: text_bytes_freq[(x[0], x[1])])
            idx = 256 + i
            
            ids = [_merge_naive(chunk_ids, freq_pair, idx) for chunk_ids in ids]
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

    def train(self, text, vocab_size, verbose=False):
        """Updates the tokenizer vocabulary based on the given text;
        performs Byte-Pair Encoding (BPE) on the text and iteratively merges the most common pairings 
        of tokens. Calculates the most frequent pair every iteration in constant time.

        NOTE: This training method will also force prevent merges between different chunks of text. Chunks are formed by the regex pattern used by GPT-4

        Time Complexity: O(2mn) where m is the length of the text and n is the number of compression iterations

        Parameters
        ----------
        text : str
            The text to train the tokenizer
        vocab_size : int
            The number of iterations to perform a merging of the most frequent pair
        verbose : bool, optional
            When enabled, will print the result of every merging of a common pair, including its frequency and its new token replacement
        """
        # Split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # Input text preprocessing
        # Convert the string to a list of utf-8 bytes
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # Store the resulting vocab table and merge table in temporary variables
        tmp_vocab = self._vocab.copy()
        tmp_merges = self._merges.copy()

        print(f"Training on {len(ids)} chunks of text...")
        
        #loader.update(n=len(tmp_vocab))
        index_i = 0
        # BPE: Iteratively merge the most common consecutive pairings of bytes
        self._mergeable_ranks = {}
        for chunk_ids in ids:
            self._mergeable_ranks = _pair_freq(chunk_ids, self._mergeable_ranks)

        loader = tqdm(total=(vocab_size - 256), desc="Training BPE", unit="merges")
        
        while index_i < vocab_size - 256:
            # Store the most freq pair
            freq_pair = max(self._mergeable_ranks, key=lambda x: self._mergeable_ranks[(x[0], x[1])])
            if self._mergeable_ranks[freq_pair] == 1:
                break

            idx = 256 + index_i
            ids = [_merge(self._mergeable_ranks, chunk_ids, freq_pair, idx) for chunk_ids in ids]
            
            tmp_merges[freq_pair] = idx
            tmp_vocab[idx] = tmp_vocab[freq_pair[0]] + tmp_vocab[freq_pair[1]]
                

            '''if verbose:
                print(f"merge {i+1}/{num_merges}: {freq_pair} -> {idx} ({tmp_vocab[idx]}) had {self._mergeable_ranks[freq_pair]} occurrences")'''

            
            index_i += 1
            loader.update()

        loader.close()

        # Save instance variables
        self._merges = tmp_merges # used in encode()
        self._vocab = tmp_vocab   # used in decode()

    def _encode_chunk(self, text_bytes):
        """Helper function for the encode() method: Returns a list of tokens (integers) encoded from the given raw bytes
        
        Parameters
        ----------
        text_bytes : bytes
            utf-8 bytes to be encoded to tokens
        """
        ids = list(text_bytes)
        while len(ids) >= 2:
            freq = _pair_freq(ids)
            recent_pair = min(freq, key=lambda x: self._merges.get(x, float("inf")))
            if recent_pair not in self._merges:
                break

            idx = self._merges[recent_pair]
            ids = _merge_naive(ids, recent_pair, idx)

        return ids
    
    def encode(self, text):
        """Returns a list of tokens given an input string

        Parameters
        ----------
        text : str
            Plain text that is to be converted to a list of tokens
        """
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

    def decode(self, ids):
        """Returns a string representation of a given list of tokens

        Parameters
        ----------
        ids : list[int]
            A sequence of tokens to be converted to text
        """
        tokens = b"".join(self._vocab[i] for i in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    def save(self, file_prefix):
        """Saves the instance's vocabulary and merge history to a text file ending in .model with the given file prefix
        
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only

        Parameters:
        file_prefix : str
            The intended file name excluding the .model suffix
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
        """Loads the vocabulary and merge history from a text file given a string; 
        model_file must include file suffix .model

        Inverse of save() but only for the model file

        Parameters
        ----------
        model_file : str
            The desired file's file name to be loaded 
        """
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

# This is the command to download the umb website
# wget -m -k -K -E -l 7 -t 6 -w 5 https://www.umb.edu/
