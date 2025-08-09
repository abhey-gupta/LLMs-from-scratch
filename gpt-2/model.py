import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import cast

@dataclass
class GPTConfig:
    window_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072
    device: torch.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class GPT2(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.window_size, config.d_model),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.d_model)
        ))
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # weight sharing between the token embedding and the output layer
        self.transformer['wte'].weight = self.lm_head.weight

        # initializing the weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, 'SCALE_WEIGHTS'):
            std *= (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.shape

        pos = torch.arange(0, T, device=device, dtype=torch.long)

        tok_emb = self.transformer['wte'](idx)
        pos_emb = self.transformer['wpe'](pos)

        x = tok_emb + pos_emb

        for block in cast(nn.ModuleList, self.transformer['h']):
            x = block(x)
        
        x = self.transformer['ln_f'](x)
        
        if targets is not None:
            # calculate loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference mode
            logits = self.lm_head(x)
            loss = None

        return logits, loss

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = MultiheadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ff = FFN(d_model=config.d_model, d_ff=config.d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_dim = config.d_model // config.n_heads
        
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj.SCALE_WEIGHTS = torch.tensor(1)

        self.register_buffer("mask", torch.tril(torch.ones(config.window_size, config.window_size)).view(1, 1, config.window_size, config.window_size))

    def forward(self, x):
        B, T, C = x.shape
        h = self.config.n_heads

        q = self.query(x) # (B, T, C) @ (C, C) => (B, T, C)
        k = self.key(x)
        v = self.value(x)

        q = q.view(B, T, h, self.head_dim).transpose(1, 2) # (B, T, C) -> (B, T, h, h_d) -> (B, h, T, h_d)
        k = k.view(B, T, h, self.head_dim).transpose(1, 2)
        v = v.view(B, T, h, self.head_dim).transpose(1, 2)


        # mask = self.get_buffer("mask")[:, :, :T, :T]
        # attn_scores = (q @ k.transpose(-2, -1)) / (k.shape[-1]**0.5) # (B, h, T, T)
        # attn_scores = attn_scores.masked_fill(mask==0, float('-inf'))
        # attn_probs = F.softmax(attn_scores, dim=-1) # (B, h, T, T)
        # attn_output = attn_probs @ v # (B, h, T, h_d)
        # Replace above with below

        # uses FlashAttention (faster computation due to fused kernels, etc)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        out = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
    
class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.c_fc = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(d_ff, d_model)
        self.c_proj.SCALE_WEIGHTS = torch.tensor(1)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)