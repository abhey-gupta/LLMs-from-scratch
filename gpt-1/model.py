import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.word_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Learned position embeddings instead of sinusoidal
        self.pos_emb = nn.Embedding(config.window_size, config.d_model)
        
        self.decoder = nn.ModuleList([Decoder(config) for _ in range(config.num_decoders)])
        self.dropout = nn.Dropout(config.dropout)

        nn.init.normal_(self.word_emb.weight, 0, 0.02)

    def forward(self, x):
        batch, window = x.shape
        positions = torch.arange(0, window, device=x.device).expand(batch, window)

        dec_out = self.dropout(self.word_emb(x) + self.pos_emb(positions))

        for layer in self.decoder:
            dec_out = layer(dec_out)

        return dec_out

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.attn = MultiheadAttention(config)
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        
        self.FF = nn.Sequential(
            nn.Linear(config.d_model, config.d_ffn),
            nn.GELU(),
            nn.Linear(config.d_ffn, config.d_model),
            nn.Dropout(config.dropout)
        )
        self.ff_norm = nn.LayerNorm(config.d_model)

    def forward(self, x):
        x = x + self.attn_dropout(self.attn(x))
        x = self.attn_norm(x)
        
        x = x + self.FF(x)
        x = self.ff_norm(x)
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_dim = config.d_model // config.heads
        
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x):
        B, T, C = x.shape
        h = self.config.heads

        q = self.query(x) # (B, T, C) @ (C, C) => (B, T, C)
        k = self.key(x)
        v = self.value(x)

        # (B, T, C) -> (B, T, h, h_d) -> (B, h, T, h_d)
        q = q.view(B, T, h, self.head_dim).transpose(1, 2)
        k = k.view(B, T, h, self.head_dim).transpose(1, 2)
        v = v.view(B, T, h, self.head_dim).transpose(1, 2)

        d_k = k.shape[-1]

        attn_scores = (q @ k.transpose(-2, -1)) / (d_k**0.5) # (B, h, T, T)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask==0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1) # (B, h, T, T)
        attn_output = attn_probs @ v # (B, h, T, h_d)
        
        out = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

# Head for language modeling
class LMHead(nn.Module):
    def __init__(self, config, model):
        super().__init__()

        self.model = model
        self.prediction = nn.Linear(config.d_model, config.vocab_size)
        self.prediction.weight = model.word_emb.weight

    def forward(self, x: torch.Tensor):
        dec_out = self.model(x)
        logits = self.prediction(dec_out)
        return logits