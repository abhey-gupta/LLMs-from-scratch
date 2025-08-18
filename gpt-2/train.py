import torch
import time
from model import ModelConfig, GPT2
from dataset import DataLoader

config = ModelConfig(vocab_size=50304)
print('Using device:', config.device)

train_loader = DataLoader('dataset.txt', B=4, T=1024)

# Uses TF32 for inner computations (Faster as precision bits are lowered)
# torch.set_float32_matmul_precision('high')

model = GPT2(config)
model.to(config.device)
# model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    optimizer.zero_grad()
    x, y = train_loader.next_batch()
    x, y = x.to(config.device), y.to(config.device)

    # Uses bfloat16 -> improves performance (as precision bits are reduced even further compared to TF32)
    # with torch.autocast(device_type="mps", dtype=torch.bfloat16):
    #     logits, loss = model(x, y)

    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    torch.mps.synchronize()
    dt = (time.time() - t0) * 1000
    print(f"Step {i+1}, Loss: {loss.item()}, dt: {dt:.2f}ms")