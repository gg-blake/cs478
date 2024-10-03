"""
Name : tokenizer.py
Description : Encodes a string of text to a series of tokens where tokens are common groupings of characters and decodes a series of tokens into text. Creates a lookup table for tokens as well.
Author : Blake Moody
Date : 9-9-2024
"""
import regex as re
from tqdm import tqdm
import unicodedata

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
        loader = tqdm(total=vocab_size)
        #loader.update(n=len(tmp_vocab))
        i = 0
        # BPE: Iteratively merge the most common consecutive pairings of bytes
        
        while len(tmp_vocab) < vocab_size:
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
    VOCAB_SIZE = 200000
    tokenizer = Tokenizer()
    tokenizer_training_data = ""
    with open("data/threebody.txt") as g:
        tokenizer_training_data += g.read()

    try:
        tokenizer.load("save4.model")
    except FileNotFoundError:
        tokenizer.train(tokenizer_training_data, VOCAB_SIZE)

    tokenizer.save("save4")
    e = tokenizer.encode("Llama is an accessible, open large language model (LLM) designed for developers, researchers, and businesses to build, experiment, and responsibly scale their generative AI ideas. Part of a foundational system, it serves as a bedrock for innovation in the global community. A few key aspects: Open access: Easy accessibility to cutting-edge large language models, fostering collaboration and advancements among developers, researchers, and organizations\nBroad ecosystem: Llama models have been downloaded hundreds of millions of times, there are thousands of community projects built on Llama and platform support is broad from cloud providers to startups - the world is building with Llama!\nTrust & safety: Llama models are part of a comprehensive approach to trust and safety, releasing models and tools that are designed to enable community collaboration and encourage the standardization of the development and usage of trust and safety tools for generative AI\nOur mission is to empower individuals and industry through this opportunity while fostering an environment of discovery and ethical AI advancements. The model weights are licensed for researchers and commercial entities, upholding the principles of openness.")
    t = [tokenizer._vocab[i] for i in e]
    print(t)