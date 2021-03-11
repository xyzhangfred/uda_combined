import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import os,sys
import random
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import BertForSequenceClassification, Trainer, TrainingArguments


class resBlock(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(resBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self,x):
        residual = x
        out1 = torch.relu(self.linear1(x))
        out2 = torch.relu(self.linear2(out1))
        out2 += residual
        return out2


class BERTProjector():
    def __init__(self, bert_model, input_dim,hidden_dim, output_dim,layer_num = 4 ):
        """
        BERT+Projector+Classifier
        """
        super().__init__()
        self.bert = bert_model
        if layer_num == 1:
            self.projector = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(input_dim, output_dim)),
        ]))
        elif layer_num == 2:
            self.projector = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(input_dim, hidden_dim)),
            ('tanh1', nn.Tanh()),
            ('linear2', nn.Linear(hidden_dim,output_dim)),
        ]))
        elif layer_num == 3:
            self.projector = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(input_dim, hidden_dim)),
            ('tanh1', nn.Tanh()),
            ('linear2', nn.Linear(hidden_dim,hidden_dim)),
            ('tanh2', nn.Tanh()),
            ('linear3', nn.Linear(hidden_dim,output_dim)),
        ]))
        elif layer_num == 4:
            self.projector = nn.Sequential(OrderedDict([
                ('linear1', nn.Linear(input_dim, hidden_dim)),
                ('tanh1', nn.Tanh()),
                ('linear2', nn.Linear(hidden_dim,hidden_dim)),
                ('tanh2', nn.Tanh()),
                ('linear3', nn.Linear(hidden_dim,hidden_dim)),
                ('tanh3', nn.Tanh()),
                ('linear4', nn.Linear(hidden_dim,output_dim))
            ]))
        elif layer_num == 5:
            self.projector = nn.Sequential(OrderedDict([
                ('linear1', nn.Linear(input_dim, hidden_dim)),
                ('tanh1', nn.Tanh()),
                ('linear2', nn.Linear(hidden_dim,hidden_dim)),
                ('tanh2', nn.Tanh()),
                ('linear3', nn.Linear(hidden_dim,hidden_dim)),
                ('tanh3', nn.Tanh()),
                ('linear4', nn.Linear(hidden_dim,hidden_dim)),
                ('tanh4', nn.Tanh()),
                ('linear5', nn.Linear(hidden_dim,output_dim))
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


class BERTResProjector():
    def __init__(self, bert_model, input_dim,hidden_dim, output_dim,block_num = 2 ):
        """
        BERT+Projector+Classifier
        """
        super().__init__()
        self.bert = bert_model
        if block_num == 1:
            self.projector = nn.Sequential(OrderedDict([
            ('block1', resBlock(input_dim,hidden_dim, output_dim)),
        ]))
        elif block_num == 2:
            self.projector = nn.Sequential(OrderedDict([
            ('block1', resBlock(input_dim,hidden_dim, output_dim)),
            ('block2', resBlock(input_dim,hidden_dim, output_dim)),
        ]))
        elif block_num == 3:
            self.projector = nn.Sequential(OrderedDict([
            ('block1', resBlock(input_dim,hidden_dim, hidden_dim)),
            ('block2', resBlock(hidden_dim,hidden_dim, hidden_dim)),
            ('block3', resBlock(hidden_dim,hidden_dim, output_dim)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(input_dim, hidden_dim)),
            ('relu', nn.ReLU()),
            ('linear2', nn.Linear(hidden_dim, 2)),
        ]))





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
