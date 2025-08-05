# GPT-1 from Scratch (Character-Level)

This is a beginner-friendly, from-scratch implementation of GPT-1, inspired by the original paper “Improving Language Understanding by Generative Pre-Training”.

It’s implemented using PyTorch and currently trains a character-level language model using a basic integer-based tokenizer. The entire model pipeline—from tokenizer to training loop and text generation—is modular, easy to follow, and beginner-ready.

## Features

- Clean and readable PyTorch code
- Character-level modeling
- Integer-based tokenizer (no external libraries required)
- Modular design: model.py, train.py, generate.py, config.py, tokenizer.py
- Fully self-contained training and generation scripts
- Checkpoint saving and resuming
- Easy to extend to word-level or BPE tokenization in the future

## Architecture Overview

This model replicates the original GPT-1 transformer decoder architecture:

- Input Embeddings: Word + learned positional embeddings
- Transformer Decoder Blocks: Multi-head masked self-attention + feedforward layers
- Layer Normalization and Residual Connections
- Language Modeling Head for next character prediction

## Tokenizer

Currently, the model uses a character-level integer tokenizer:

- Each character is mapped to a unique integer (stoi)
- Reverse lookup is supported via (itos)
- Future versions will include support for subword or word-level tokenization

## Project Structure

```
├── model.py        # GPT-1 model with decoder, attention, and LM head
├── tokenizer.py    # Character-level tokenizer
├── config.py       # Model/training configuration
├── dataset.py      # Dataset loading and preparation
├── train.py        # Training loop with loss logging and checkpointing
├── generate.py     # Text generation from a prompt
├── checkpoint.pt   # Auto-saved during training
├── data.txt        # Input text file for training
```

## Requirements

- Python 3.8+
- PyTorch (MPS, CUDA or CPU supported)

Install dependencies:

```
pip install torch
```

## Getting Started

### 1. Prepare the data

Put your training corpus in a file named `data.txt`.

For example:

```
To be, or not to be, that is the question...
```

### 2. Train the model

```
python train.py
```

This will train the model from scratch and periodically save checkpoints to `checkpoint.pt`.

### 3. Generate text

```
python generate.py
```

You'll be prompted to enter a starting string. The model will generate additional characters based on the prompt.

## Example

```
Enter the starting text: Once upon a time
Once upon a time, the king said, "We shall go to war!"
```

## Configuration

Adjust model size and hyperparameters in `config.py`:

```python
class Config:
    d_model = 384
    heads = 6
    num_decoders = 6
    d_ffn = 1536
    dropout = 0.2
    window_size = 256
    batch_size = 64
    vocab_size = 5000
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

## License

This project is open-source and MIT licensed. Use it freely for learning or extension.