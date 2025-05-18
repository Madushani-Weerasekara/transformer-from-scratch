import torch # A Python library for building neural networks 
import  torch.nn as nn #  To build layers for neural networks
import math # To perform math functions

class ImputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__() # Initialize parent class (nn.Module
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

