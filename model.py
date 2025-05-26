import torch # A Python library for building neural networks 
import  torch.nn as nn #  To build layers for neural networks
import math # To perform math functions

class ImputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__() # Initialize parent class (nn.Module)
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Create Embedding Layer
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout:float) -> None:
        super().__init()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape(seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of shape(seq_len, 1)
        position = torch.arrange(0, seq_len, d_type=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arrange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        #Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the cosine to the ood poaitions
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add the new dimension
        pe = pe.unsqueeze(0) # It will become a tensor of seq_len to d_model

        self.register_buffer('pe', pe)

    def forwad(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # requires_grad_(False) means we don't want to learn the positions during the training process, they are just fixed
        return self.dropout(x)
    





