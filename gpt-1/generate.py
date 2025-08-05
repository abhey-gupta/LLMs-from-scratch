import os
import torch
import torch.nn.functional as F
from model import GPT, LMHead
from config import Config
from dataset import Dataset

dataset = Dataset("data.txt")
config = Config()
config.vocab_size = dataset.tokenizer.vocab_size()

gpt = GPT(config).to(config.device)
lm = LMHead(config, gpt).to(config.device)

# Load checkpoint
checkpoint_path = "checkpoint.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    lm.load_state_dict(checkpoint["model_state"])
    print(f"Loaded model from checkpoint at iter {checkpoint['iter']}")
else:
    raise FileNotFoundError("No checkpoint found.")

lm.eval()

def generate(lm, start_text, max_new_tokens=1000, temperature=1.0):
    context = torch.tensor(dataset.tokenizer.encode(start_text), dtype=torch.long, device=config.device).unsqueeze(0)

    generated = context

    # print the prompt
    print(start_text, end="", flush=True)

    for _ in range(max_new_tokens):
        context_crop = generated[:, -config.window_size:]

        with torch.no_grad():
            logits = lm(context_crop)

        next_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat((generated, next_token), dim=1)

        # print the generated character
        next_char = dataset.tokenizer.decode(next_token[0].tolist())
        print(next_char, end="", flush=True)

    print()
    out = generated[0].tolist()
    return dataset.tokenizer.decode(out)

if __name__ == "__main__":
    prompt = input("Enter the starting text: ")
    output = generate(lm, prompt, max_new_tokens=1000)
    print("\nGenerated:\n")
    print(output)