import torch
import torch.nn as nn


class PromptEncoder(torch.nn.Module):
    def __init__(self, prompt_num, hidden_size, device, dropout_prob):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        # ent embedding

        self.seq_indices = torch.LongTensor(list(range(prompt_num))).to(self.device)
        # embedding
        self.embedding = torch.nn.Embedding(prompt_num, self.hidden_size)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=dropout_prob,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds # [10,768]
