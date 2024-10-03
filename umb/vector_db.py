from tokenizer import Tokenizer
import numpy as np
import os
import torch
from graphviz import Graph
import networkx as nx
import torch.nn.functional as F

class VectorDB:
    def __init__(self, tokenizer, embedding_size):
        self.tokenizer = tokenizer
        self.embedding_size = embedding_size
        self._vectors = torch.zeros((0, embedding_size))
        self._vectors_index = Graph()

    def train_from_document(self, file_paths):
        cat_text = ""
        for file_path in file_paths:
            with open(file_path) as f:
                cat_text += f.read()

        self.tokenizer.train(cat_text, self.embedding_size)
        for file_path in file_paths:
            with open(file_path) as f:
                text = f.read()
                self.tokenizer.encode(text)
                tokens = torch.tensor(self.tokenizer.encode(text))
                token_freq = self._token_frequency(tokens)
                self.add_vector(token_freq)

    def _token_frequency(self, tokens):
        token_freq = torch.zeros(len(self.tokenizer._vocab))
        for token in tokens:
            token_freq[token] += 1

        return token_freq.softmax(dim=0)
    
    def add_vector(self, vector):
        # Add the vector to the database (vector should be normalized with softmax)
        self._vectors_index.node(str(len(self._vectors)))
        if len(self._vectors) == 0:
            self._vectors = torch.cat((self._vectors, vector.unsqueeze(0)))
            return
        
        similarity_vector = F.cosine_similarity(vector, self._vectors, dim=1)
        for i, similarity in enumerate(similarity_vector):
            self._vectors_index.edge(str(len(self._vectors)), str(i), label=str(similarity.item()))

        self._vectors = torch.cat((self._vectors, vector.unsqueeze(0)))

    def find_similar_text(self, text, number_of_results=5):
        vector = self._token_frequency(torch.tensor(self.tokenizer.encode(text)))

        # Find the most similar vectors in the database
        similarity_vector = F.cosine_similarity(vector, self._vectors, dim=1)
        similarity_vector = similarity_vector.argsort(descending=True)
        return similarity_vector[:number_of_results]
    
    def visualize(self):
        self._vectors_index.render("graph.png")

if __name__ == "__main__":
    tokenizer = Tokenizer()
    db = VectorDB(tokenizer, 3000)
    db.train_from_document(["data/d1.txt", "data/d2.txt", "data/d3.txt"])
    db.visualize()
    search_text = "Financial aid and the bursars office and student loans"
    print(db._vectors)

    



        
        

    

    
       