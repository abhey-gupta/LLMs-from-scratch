import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import cast

@dataclass
class ModelConfig:
    size: str = "small"
    window_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072
    device: torch.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    def __post_init__(self):
        size_map = {
            "small":  {"n_layer": 12, "n_heads": 12, "d_model": 768,  "d_ff": 3072},
            "medium": {"n_layer": 24, "n_heads": 16, "d_model": 1024, "d_ff": 4096},
            "large":  {"n_layer": 36, "n_heads": 20, "d_model": 1280, "d_ff": 5120},
            "xl":     {"n_layer": 48, "n_heads": 25, "d_model": 1600, "d_ff": 6400},
        }

        if self.size not in size_map:
            raise ValueError(f"Invalid size '{self.size}'. Choose from {list(size_map.keys())}.")

        cfg = size_map[self.size]
        self.n_layer = cfg["n_layer"]
        self.n_heads = cfg["n_heads"]
        self.d_model = cfg["d_model"]
        self.d_ff = cfg["d_ff"]

class GPT2(nn.Module):
    def __init__(self, config: ModelConfig):
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
    
    @classmethod
    def from_pretrained(cls, model_size='124M'):
        from transformers import GPT2LMHeadModel

        size_map = {
            '124M': 'gpt2',
            '355M': 'gpt2-medium',
            '774M': 'gpt2-large',
            '1558M': 'gpt2-xl'
        }
        if model_size not in size_map:
            raise ValueError(f"Invalid size {model_size}. Choose from {list(size_map.keys())}")

        hf_model = GPT2LMHeadModel.from_pretrained(size_map[model_size])
        hf_cfg = hf_model.config
        hf_state = hf_model.state_dict()

        config = ModelConfig(
            window_size=hf_cfg.n_positions,
            vocab_size=hf_cfg.vocab_size,
            n_layer=hf_cfg.n_layer,
            n_heads=hf_cfg.n_head,
            d_model=hf_cfg.n_embd,
            d_ff=hf_cfg.n_inner if hf_cfg.n_inner is not None else 4 * hf_cfg.n_embd
        )
        model = cls(config)
        own_sd = model.state_dict()

        # Copy token & position embeddings
        own_sd["transformer.wte.weight"] = hf_state["transformer.wte.weight"]
        own_sd["transformer.wpe.weight"] = hf_state["transformer.wpe.weight"]

        # Loop over each block
        for layer in range(config.n_layer):
            # LayerNorms
            for ln in ["ln_1", "ln_2"]:
                own_sd[f"transformer.h.{layer}.{ln}.weight"] = hf_state[f"transformer.h.{layer}.{ln}.weight"]
                own_sd[f"transformer.h.{layer}.{ln}.bias"] = hf_state[f"transformer.h.{layer}.{ln}.bias"]

            # Attention Q, K, V
            c_attn_w = hf_state[f"transformer.h.{layer}.attn.c_attn.weight"]
            c_attn_b = hf_state[f"transformer.h.{layer}.attn.c_attn.bias"]
            d = config.d_model
            own_sd[f"transformer.h.{layer}.attn.query.weight"] = c_attn_w[:, :d].T
            own_sd[f"transformer.h.{layer}.attn.key.weight"]   = c_attn_w[:, d:2*d].T
            own_sd[f"transformer.h.{layer}.attn.value.weight"] = c_attn_w[:, 2*d:].T
            own_sd[f"transformer.h.{layer}.attn.query.bias"] = c_attn_b[:d]
            own_sd[f"transformer.h.{layer}.attn.key.bias"]   = c_attn_b[d:2*d]
            own_sd[f"transformer.h.{layer}.attn.value.bias"] = c_attn_b[2*d:]

            # Attention output projection
            own_sd[f"transformer.h.{layer}.attn.out_proj.weight"] = hf_state[f"transformer.h.{layer}.attn.c_proj.weight"].T
            own_sd[f"transformer.h.{layer}.attn.out_proj.bias"]   = hf_state[f"transformer.h.{layer}.attn.c_proj.bias"]

            # FFN
            own_sd[f"transformer.h.{layer}.ff.c_fc.weight"] = hf_state[f"transformer.h.{layer}.mlp.c_fc.weight"].T
            own_sd[f"transformer.h.{layer}.ff.c_fc.bias"]   = hf_state[f"transformer.h.{layer}.mlp.c_fc.bias"]
            own_sd[f"transformer.h.{layer}.ff.c_proj.weight"] = hf_state[f"transformer.h.{layer}.mlp.c_proj.weight"].T
            own_sd[f"transformer.h.{layer}.ff.c_proj.bias"]   = hf_state[f"transformer.h.{layer}.mlp.c_proj.bias"]

        # Final layer norm
        own_sd["transformer.ln_f.weight"] = hf_state["transformer.ln_f.weight"]
        own_sd["transformer.ln_f.bias"]   = hf_state["transformer.ln_f.bias"]

        # LM Head
        own_sd["lm_head.weight"] = hf_state["lm_head.weight"]

        model.load_state_dict(own_sd)
        return model

    def generate(self, prompt: str, max_tokens: int = 50, eos_token_id: int = 50256, temperature=1.0):
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        self.eval()
        device = next(self.parameters()).device

        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        print(prompt, end="")
        # Autoregressive loop
        for _ in range(max_tokens):
            input_ids = input_ids[:, -self.config.window_size:]
            logits, _ = self(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == eos_token_id:
                break
            print(tokenizer.decode(next_token.item()), end="", flush=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids[0]

class Block(nn.Module):
    def __init__(self, config: ModelConfig):
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
    

if __name__ == "__main__":
    config = ModelConfig()
    print('loaded config')
    model = GPT2.from_pretrained('355M').to(config.device)
    print('loaded model')
    model.generate(
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nIdentify the correct spelling of the following word.\n\n ### Input:\nOcassion", 
        max_tokens=100)