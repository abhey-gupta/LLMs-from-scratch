from tokenizer import CharTokenizer

class Dataset:
    def __init__(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            self.text = f.read()
        self.tokenizer = CharTokenizer(self.text)

    def get_text(self):
        return self.text

    def get_tokenizer(self):
        return self.tokenizer