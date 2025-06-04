import torch
import torch.nn as nn
from transformers import BertModel

class BertRegressor(nn.Module):
    def __init__(self, hidden_size=1024, dropout=0.2):
        super(BertRegressor, self).__init__()
        self.bert = BertModel.from_pretrained("Rostlab/prot_bert", output_hidden_states=True)
        # Descongelar las últimas 4 capas de BERT
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-4:].parameters():
            param.requires_grad = True
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        mean_pooled = torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return self.regressor(mean_pooled).squeeze(-1)