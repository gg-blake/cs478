import torch
from transformers import BertTokenizer, BertModel
import nltk

nltk.download('punkt_tab')  # Download the Punkt sentence tokenizer if not already present
from nltk.tokenize import sent_tokenize

text = "This is the first sentence. This is the second. And this is the third. And this is the fourth. And this is the fifth."
sentences = sent_tokenize(text)
print(sentences)

# For more details - https://huggingface.co/bert-base-uncased
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokens = tokenizer(*sent_tokenize(text))
print(tokenizer.decode(tokens.labels))