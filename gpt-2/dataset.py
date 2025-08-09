import torch
import tiktoken

class DataLoader:
    def __init__(self, path, B, T):
        self.B, self.T = B, T

        with open(path, 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.current_position = 0
        print(f"Total tokens: {len(self.tokens)}")
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T
        if self.current_position + (B * T + 1) >= len(self.tokens):
            self.current_position = 0
        return x, y