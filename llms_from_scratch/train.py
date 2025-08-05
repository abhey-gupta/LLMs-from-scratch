import os
import torch
import torch.nn.functional as F
from dataset import Dataset
from model import GPT 
from model import LMHead
from config import Config

dataset = Dataset("data.txt")

data = torch.tensor(dataset.tokenizer.encode(dataset.get_text()), dtype=torch.long)
vocab_size = dataset.tokenizer.vocab_size()
print(f"Dataset has {len(data)} tokens, vocab size: {vocab_size}")

# Get dataset splits
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

config = Config()
config.vocab_size = vocab_size
context_len = config.window_size
print(f"Using device: {config.device}")

# Model and optimizer
gpt = GPT(config).to(config.device)
lm = LMHead(config, gpt).to(config.device)
print(f"No of parameters: {sum(p.numel() for p in lm.parameters())}")
optimizer = torch.optim.AdamW(lm.parameters(), lr=3e-4)

# Checkpoint
checkpoint_path = "checkpoint.pt"
start_iter = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    lm.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_iter = checkpoint["iter"]
    print(f"Resumed from checkpoint at iter {start_iter}")

# Batch function
def get_batch(split="train"):
    source = train_data if split == "train" else val_data
    ix = torch.randint(0, len(source) - context_len - 1, (config.batch_size,))
    x = torch.stack([source[i:i+context_len] for i in ix])
    y = torch.stack([source[i+1:i+context_len+1] for i in ix])
    return x.to(config.device), y.to(config.device)

# Validation estimator
@torch.no_grad()
def estimate_loss(eval_iters=10):
    lm.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            logits = lm(xb)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), yb.view(B*T))
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    lm.train()
    return out

# Training loop
num_iters = 5000
log_interval = 100

for iter in range(start_iter, num_iters):
    lm.train()
    x, y = get_batch("train")
    logits = lm(x)

    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % log_interval == 0:
        losses = estimate_loss()
        print(f"Iter {iter}: train loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}")

        torch.save({
            "iter": iter,
            "model_state": lm.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, checkpoint_path)