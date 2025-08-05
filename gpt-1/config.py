import torch

# used in the original GPT-1 model
# class Config:
#     vocab_size: int = 5000
#     d_model: int = 768
#     heads: int = 12
#     d_ffn: int = 3072
#     num_decoders: int = 12
#     window_size: int = 256

class Config:
    d_model: int = 384
    batch_size: int = 64
    window_size: int = 256
    vocab_size: int = 5000
    heads: int = 6
    d_ffn: int = 1536
    num_decoders: int = 6
    dropout: float = 0.2
    device: torch.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')