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
    

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        # This make parameter leanerble
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied 
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True) # Calculate mean
        std = x.std(dim = -1, keepdim=True) # Calculate Standard deviation
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W1 B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_ff) #W2 B2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttentionBlok(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv
        
        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1] # d_k is the last dimention of the Q, K and V

        # (Batch, h, seq_len, d_k) --> (Bath, h, seq_len, seq_len )
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Before applying the softmax we need to apply the mask
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)
        # Now we will apply softmax
        attention_scores = attention_scores.softmax(dim=1) # (Batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Finally we multiply the output of the softmax by V matrix
        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)

        # Now we ned to divide these query, key and values to smaller matrices to different heads
        # (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, h, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_score = MultiHeadAttentionBlok.attention(query, key, value, mask, self.dropout)
        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)
    
class ResidulConnection(nn.Module):
    def __init__(self, droptout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(droptout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlok, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Now we define the two residual connections
        self.residual_connections = nn.ModuleList([ResidulConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # 1st Residual connection
        x = self.residual_connections[1](self.feed_forward_block)
        return x
        
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers: # Apply one laye after another
            x = layer(x, mask)
        return self.norm(x) # Apply the normlaization

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlok, cross_attention_block: MultiHeadAttentionBlok, feed_forward_block: FeedForwardBlock, dropout: float) ->None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # Define Residual connections. In this case we have 3 of them
        self.residual_connections = nn.ModuleList([ResidulConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_msk):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_msk)) # 1st Residual connection(Self-Attention Block of the Decoder)
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)) # 2nd Residual connection(Cross-Attention Block)
        x = self.residual_connections[2](x, self.feed_forward_block) # 3rd Residual connection(Feed Forward Block)
        return x
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask) # Each layer is a Decoder Block
        return self.norm

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, seq_len, vocab_size) --> (Batch, seq_len, vocab_seze)
        return torch.log_softmax(self.proj(x), dim=-1)