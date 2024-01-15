import torch
from torch import nn 
from torch.nn import functional as F 
from attention import SelfAttention 

class CLIPEmbedding(nn.Module):
    """
    Embedding for CLIP model
    Args:
        tokens: torch.Tensor, (batch_size, seq_len)
    Returns:
        x: torch.Tensor, (batch_size, seq_len, embed_dim)
    """

    def __init__(self, n_vocab, embed_dim, seq_len):
        super().__init__()

        self.tokeembed_dimding = nn.Embedding(n_vocab, embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(seq_len, embed_dim))
    
    def forward(self, tokens):
        # (batch_size, seq_len)
        x = self.tokeembed_dimding(tokens)
        # (batch_size, seq_len, embed_dim)
        x = x + self.positional_embedding
        # (batch_size, seq_len, embed_dim)
        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_heads, embed_dim):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.self_attention = SelfAttention(n_heads, embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear2 = nn.Linear(4 * embed_dim, embed_dim)
    
    def forward(self, x):
        residual = x
        # (batch_size, seq_len, embed_dim)
        x = self.layernorm1(x)
        # (batch_size, seq_len, embed_dim)
        x = self.self_attention(x, causal_mask=True)
        # (batch_size, seq_len, embed_dim)
        x = residual + x
        # (batch_size, seq_len, embed_dim)
        residual = x
        # (batch_size, seq_len, embed_dim)
        x = self.layernorm2(x)
        # (batch_size, seq_len, embed_dim)
        x = self.linear1(x)
        # (batch_size, seq_len, embed_dim)
        x = x * torch.sigmoid(1.702 * x) # swish
        # (batch_size, seq_len, embed_dim)
        x = self.linear2(x)
        # (batch_size, seq_len, embed_dim)
        x = residual + x
        # (batch_size, seq_len, embed_dim)
        return x

class CLIP(nn.Module):
    def __init__(self, n_vocab, embed_dim, seq_len, n_heads, n_layers):
        super().__init__()

        self.embedding = CLIPEmbedding(n_vocab, embed_dim, seq_len)
        self.layers = nn.ModuleList([CLIPLayer(n_heads, embed_dim) for _ in range(n_layers)])
        self.layernorm = nn.LayerNorm(embed_dim)
    
    def forward(self, tokens):
        tokens = tokens.type(torch.long)
        # (batch_size, seq_len)
        x = self.embedding(tokens)
        # (batch_size, seq_len, embed_dim)
        for layer in self.layers:
            x = layer(x)
            # (batch_size, seq_len, embed_dim)
        x = self.layernorm(x)
        # (batch_size, seq_len, embed_dim)
        return x



