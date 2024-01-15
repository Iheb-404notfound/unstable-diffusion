import torch
from torch import nn
from torch.nn import functional as F 

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_model, in_bias=True, out_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=in_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=out_bias)
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
    
    def forward(self, x, casual_mask=False):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape

        x = self.in_proj(x)
        # (batch_size, seq_len, 3 * d_model)
        q, k, v = x.chunk(3, dim=-1)
        # 3 * (batch_size, seq_len, d_model)
        sh = (batch_size, seq_len, self.n_heads, self.d_head)
        q = q.view(sh).transpose(1, 2)
        k = k.view(sh).transpose(1, 2)
        v = v.view(sh).transpose(1, 2)
        # (batch_size, n_heads, seq_len, d_head)
        attention_weights = q @ k.transpose(-1, -2)
        # (batch_size, n_heads, seq_len, seq_len)

        if casual_mask:
            mask = torch.ones_like(attention_weights, dtype=torch.bool).triu(1)
            attention_weights.masked_fill_(mask, -torch.inf)
        # (batch_size, n_heads, seq_len, seq_len)
        
        attention_weights /= torch.sqrt(self.d_head)
        # (batch_size, n_heads, seq_len, seq_len)
        attention_weights = F.softmax(attention_weights, dim=-1)
        # (batch_size, n_heads, seq_len, seq_len)
        out = attention_weights @ v
        # (batch_size, n_heads, seq_len, d_head)
        out = out.reshape((batch_size, seq_len, d_model))
        # (batch_size, seq_len, d_model)
        out = self.out_proj(out)
        # (batch_size, seq_len, d_model)
        return out


class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_cross, in_bias=True, out_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=in_bias)
        self.k_proj = nn.Linear(d_cross, d_model, bias=in_bias)
        self.v_proj = nn.Linear(d_cross, d_model, bias=in_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=out_bias)
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
    
    def forward(self, x, y):
        # x: (batch_size, sequ_len_q, d_model_q)
        # y: (batch_size, sequ_len_kv, d_model_kv)
        batch_size, seq_len_q, d_model = x.shape
        _, seq_len_kv, d_cross = y.shape

        sh = (batch_size, -1, self.n_heads, self.d_head)

        # (batch_size, seq_len_q, d_model_q)
        q = self.q_proj(x)
        # (batch_size, seq_len_q, d_model_q)

        # (batch_size, seq_len_kv, d_model_kv)
        k = self.k_proj(y)
        v = self.v_proj(y)
        # (batch_size, seq_len_kv, d_model_q)

        q = q.view(sh).transpose(1, 2)
        # (batch_size, n_heads, seq_len_q, d_head)
        k = k.view(sh).transpose(1, 2)
        v = v.view(sh).transpose(1, 2)
        # (batch_size, n_heads, seq_len_kv, d_head)

        attention_weights = q @ k.transpose(-1, -2)
        # (batch_size, n_heads, seq_len_q, seq_len_kv)
        attention_weights /= torch.sqrt(self.d_head)
        # (batch_size, n_heads, seq_len_q, seq_len_kv)
        attention_weights = F.softmax(attention_weights, dim=-1)
        # (batch_size, n_heads, seq_len_q, seq_len_kv)
        out = attention_weights @ v
        # (batch_size, n_heads, seq_len_q, d_head)
        out = out.transpose(1, 2).contiguous()
        # (batch_size, seq_len_q, n_heads, d_head)
        out = out.view((batch_size, seq_len_q, d_model))
        # (batch_size, seq_len_q, d_model)
        out = self.out_proj(out)
        # (batch_size, seq_len_q, d_model)
        return out
