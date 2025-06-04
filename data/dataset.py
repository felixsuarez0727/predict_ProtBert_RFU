import torch
from torch.utils.data import Dataset

class PeptideDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = " ".join(list(self.sequences[idx]))  # Espaciado para tokenización ProtBert
        encoded = self.tokenizer(seq, truncation=True, padding='max_length',
                                 max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }
