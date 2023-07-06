import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

import torch.optim as optim

def get_criterion_and_optimizer(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00001)
    return criterion, optimizer

class MyModel(nn.Module):
    def __init__(self, output_size, drop_out):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(drop_out)
        self.linear = nn.Linear(in_features=768, out_features=self.output_size, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, batch_input, batch_masked):
        out = self.bert(input_ids=batch_input, attention_mask=batch_masked)
        out = out.pooler_output
        out = self.relu(out)
        out = self.drop_out(out)
        out = self.linear(out)
        probabilities = self.softmax(out)
        return probabilities

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x