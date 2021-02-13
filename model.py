import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import os,sys
import random
import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

class BERTProjector():
    def __init__(self, bert_model, input_dim,hidden_dim, output_dim):
        """
        BERT+Projector+Classifier
        """
        super().__init__()
        self.bert = bert_model
        self.projector = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(input_dim, hidden_dim)),
            ('tanh1', nn.ReLU()),
            ('linear2', nn.Linear(hidden_dim,hidden_dim)),
            ('relu2', nn.ReLU()),
            ('linear3', nn.Linear(hidden_dim,hidden_dim)),
            ('relu3', nn.ReLU()),
            ('linear4', nn.Linear(hidden_dim,output_dim))
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(input_dim, hidden_dim)),
            ('tanh', nn.Tanh()),
            ('linear2', nn.Linear(hidden_dim, 2)),
        ]))


    def project(self, dataloader, device, paired = False):
        all_outputs = []
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                batch = [t.to(device) for t in batch]
                if paired:
                    output = self.model(batch[0])
                else:
                    output = self.model(torch.stack(batch))
                all_outputs.append(output.detach().cpu().numpy())
        output = np.concatenate(all_outputs, 0)
        return output




# class Classifier(nn.Module):
#     """ Classifier with Transformer """
#     def __init__(self,hidden_dim ,n_labels):
#         super().__init__()
#         self.fc = nn.Linear(hidden_dim, hidden_dim)
#         self.activ = nn.Tanh()
#         self.drop = nn.Dropout(cfg.p_drop_hidden)
#         self.classifier = nn.Linear(cfg.dim, n_labels)

#     def forward(self, input_ids, segment_ids, input_mask):
#         h = self.transformer(input_ids, segment_ids, input_mask)
#         # only use the first h in the sequence
#         pooled_h = self.activ(self.fc(h[:, 0])) # 맨앞의 [CLS]만 뽑아내기
#         logits = self.classifier(self.drop(pooled_h))
#         return logits
