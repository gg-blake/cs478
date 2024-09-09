"""
Name : tokenizer.py
Description : Encodes a string of text to a series of tokens where tokens are common groupings of characters and decodes a series of tokens into text. Creates a lookup table for tokens as well.
Author : Blake Moody
Date : 9-9-2024
"""
import regex as re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, pattern=None):
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self._merges = {}
        self._special_tokens = {}
        self._vocab = self._build_vocab()
    
    # Helper method to initialize an instance's vocabulary
    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self._merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self._special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    # Train the tokenizer on the given text
    # NOTE: This training method will also force prevent merges between different chunks of text. Chunks are formed by the regex pattern used by GPT-4
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256

        # Only add vocab_size number of symbols to the vocabulary
        num_merges = vocab_size - 256

        # Split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # Input text preprocessing
        # Convert the string to a list of utf-8 bytes
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # Store the resulting vocab table and merge table in temporary variables
        tmp_vocab = {idx: bytes([idx]) for idx in range(256)}
        tmp_merges = {}

        # BPE: Iteratively merge the most common consecutive pairings of bytes
        for i in range(num_merges):
            # Get the frequency stats for all chunks of text
            text_bytes_freq = {}
            for chunk_ids in ids:
                text_bytes_freq = self._pair_freq(chunk_ids, text_bytes_freq)

            # Store the most freq pair
            freq_pair = max(text_bytes_freq, key=lambda x: text_bytes_freq[x])
            idx = 256 + i
            ids = [self._merge(chunk_ids, freq_pair, idx) for chunk_ids in ids]
            tmp_merges[freq_pair] = idx
            tmp_vocab[idx] = tmp_vocab[freq_pair[0]] + tmp_vocab[freq_pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {freq_pair} -> {idx} ({tmp_vocab[idx]}) had {text_bytes_freq[freq_pair]} occurrences")

        # Save instance variables
        self._merges = tmp_merges # used in encode()
        self._vocab = tmp_vocab   # used in decode()

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

    # Helper function for BPE (Byte-Pair Encoding); Returns an dictionary containing all existing consecutive character pairs as keys and their respective frequency as the value
    def _pair_freq(self, text_bytes, pair_freq_counts=None):
        pair_freq = {} if pair_freq_counts is None else pair_freq_counts
        for p0, p1 in zip(text_bytes, text_bytes[1:]):
            if (p0, p1) not in pair_freq.keys():
                pair_freq[(p0, p1)] = 1
                continue

            pair_freq[(p0, p1)] = pair_freq[(p0, p1)] + 1

        return pair_freq


if __name__ == "__main__":
    test_string = "Our discussions with The New York Times had appeared to be progressing constructively through our last communication on December 19. The negotiations focused on a high-value partnership around real-time display with attribution in ChatGPT, in which The New York Times would gain a new way to connect with their existing and new readers, and our users would gain access to their reporting. We had explained to The New York Times that, like any single source, their content didn't meaningfully contribute to the training of our existing models and also wouldn't be sufficiently impactful for future training. Their lawsuit on December 27—which we learned about by reading The New York Times—came as a surprise and disappointment to us."
    text_bytes = list(test_string.encode("utf-8"))

    tokenizer = Tokenizer()
    tokenizer.train(test_string, 270)
    encoded_string = tokenizer.encode("What's up everyone! ComedyShortsGamer here! Today we're gonna be doing the hot knife challenge!")
    print(encoded_string)
    decoded_string = tokenizer.decode(encoded_string)
    print(decoded_string)