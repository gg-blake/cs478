"""
Name : gpt-shakespeare.py
Description : 
Author : Blake Moody
Date : 9-9-2024
"""

with open("data/d2.txt") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)
