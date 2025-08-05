class CharTokenizer:
    def __init__(self, text):
        self.vocab = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, tokens):
        return ''.join([self.itos[i] for i in tokens])

    def vocab_size(self):
        return len(self.vocab)