# Copyright 2019 SanghunYun, Korea University.
# (Strongly inspired by Dong-Hyun Lee, Kakao Brain)
# 
# This file has been modified by SanghunYun, Korea Univeristy.
# Little modification at Tokenizing, AddSpecialTokensWithTruncation, TokenIndexing
# and CsvDataset, load_data has been newly written.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ast
import csv
import itertools

import pandas as pd    # only import when no need_to_preprocessing
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from .read_data import read_imdb_cf_data, get_sup_dataset, get_eval_dataset,read_imdb_contrast_data,read_matres_data
from .data_utils import get_paired_dataloader,get_single_dataloader
# from .load_data_orig import 
from transformers import BertTokenizer

class load_data:
    def __init__(self, cfg):
        #cfg contains the parameters
        self.cfg = cfg
        self.pipeline = None
        self.n_sup = cfg.n_sup
        self.data_type = cfg.data_type
        self.sup_data_dir = cfg.sup_data_dir
        self.eval_data_dir= cfg.eval_data_dir
        self.sup_batch_size = cfg.train_batch_size
        self.eval_batch_size = cfg.eval_batch_size
        self.unsup_data_dir = cfg.unsup_data_dir
        self.unsup_batch_size = cfg.train_batch_size * cfg.unsup_ratio
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.shuffle = True

    #need a function for loading dataset, both sup and unsup.
    def sup_data_iter(self):
        if self.data_type == 'imdb_cf':
            orig_sents,orig_labels,cf_sents, cf_labels = read_imdb_cf_data(prefix = 'train', max_len = 128,data_dir = self.sup_data_dir,num=200)
        elif self.data_type == 'imdb_contrast':
            #for imdb_contrast use test data as training data since there's more of that
            orig_sents,orig_labels,cf_sents, cf_labels = read_imdb_contrast_data(prefix = 'test', max_len = 128,data_dir = self.sup_data_dir,num=100)
        elif self.data_type == 'matres':
            orig_sents,orig_labels,cf_sents, cf_labels = read_matres_data(prefix = 'train', max_len = 128,data_dir = self.sup_data_dir,num=100)

        sup_data_iter = get_paired_dataloader(self.tokenizer,orig_sents,orig_labels,cf_sents, cf_labels, batch_size = 8,max_len=128)
        return sup_data_iter

    def unsup_data_iter(self):
        if 'imdb' in self.data_type:
            unsup_dataset = IMDB_dataset(self.unsup_data_dir, self.cfg.need_prepro, self.pipeline, self.cfg.max_seq_length, self.cfg.mode, 'unsup')
            unsup_data_iter = DataLoader(unsup_dataset, batch_size=self.unsup_batch_size, shuffle=self.shuffle)
            breakpoint()
        elif self.data_type == 'matres':
            unsup_dataset = matres_unsup_dataset(self.unsup_data_dir, self.cfg.max_seq_length)
            unsup_data_iter = DataLoader(unsup_dataset, batch_size=self.unsup_batch_size, shuffle=self.shuffle)

        return unsup_data_iter


    def eval_data_iter(self):
        if self.data_type == 'imdb_cf':
            orig_eval_sents,orig_eval_labels, cf_eval_sents, cf_eval_labels = read_imdb_cf_data(prefix = 'dev', data_dir = self.eval_data_dir, max_len = 50,num = 10000)
        elif self.data_type == 'imdb_contrast':
            #for imdb_contrast use test data as training data since there's more of that
            orig_eval_sents,orig_eval_labels, cf_eval_sents, cf_eval_labels = read_imdb_contrast_data(prefix = 'dev', max_len = 128,data_dir = self.sup_data_dir,num=100)
        elif self.data_type == 'matres':
            orig_eval_sents,orig_eval_labels, cf_eval_sents, cf_eval_labels = read_matres_data(prefix = 'test', max_len = 128,data_dir = self.sup_data_dir,num=100)
        orig_eval_data_iter = get_single_dataloader(self.tokenizer,orig_eval_sents,orig_eval_labels,batch_size = 8,max_len=128,shuffle=False)
        cf_eval_data_iter = get_single_dataloader(self.tokenizer,cf_eval_sents, cf_eval_labels, batch_size = 8,max_len=128,shuffle=False)

        return orig_eval_data_iter,cf_eval_data_iter

   

def load_unsup_dataset(data_dir, max_seq_length):
    
    return Dataset

class CsvDataset(Dataset):
    labels = None
    def __init__(self, file, need_prepro, pipeline, max_len, mode, d_type):
        Dataset.__init__(self)
        self.cnt = 0

        # need preprocessing
        if need_prepro:
            with open(file, 'r', encoding='utf-8') as f:
                lines = csv.reader(f, delimiter='\t', quotechar='"')

                # supervised dataset
                if d_type == 'sup':
                    # if mode == 'eval':
                        # sentences = []
                    data = []

                    for instance in self.get_sup(lines):
                        # if mode == 'eval':
                            # sentences.append([instance[1]])
                        for proc in pipeline:
                            instance = proc(instance, d_type)
                        data.append(instance)

                    self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]
                    # if mode == 'eval':
                        # self.tensors.append(sentences)

                # unsupervised dataset
                elif d_type == 'unsup':
                    data = {'ori':[], 'aug':[]}
                    for ori, aug in self.get_unsup(lines):
                        for proc in pipeline:
                            ori = proc(ori, d_type)
                            aug = proc(aug, d_type)
                        self.cnt += 1
                        # if self.cnt == 10:
                            # break
                        data['ori'].append(ori)    # drop label_id
                        data['aug'].append(aug)    # drop label_id
                    ori_tensor = [torch.tensor(x, dtype=torch.long) for x in zip(*data['ori'])]
                    aug_tensor = [torch.tensor(x, dtype=torch.long) for x in zip(*data['aug'])]
                    self.tensors = ori_tensor + aug_tensor
        # already preprocessed
        else:
            f = open(file, 'r', encoding='utf-8')
            data = pd.read_csv(f, sep='\t')

            # supervised dataset
            if d_type == 'sup':
                # input_ids, segment_ids(input_type_ids), input_mask, input_label
                input_columns = ['input_ids', 'input_type_ids', 'input_mask', 'label_ids']
                self.tensors = [torch.tensor(data[c].apply(lambda x: ast.literal_eval(x)), dtype=torch.long)    \
                                                                                for c in input_columns[:-1]]
                self.tensors.append(torch.tensor(data[input_columns[-1]], dtype=torch.long))
                
            # unsupervised dataset
            elif d_type == 'unsup':
                input_columns = ['ori_input_ids', 'ori_input_type_ids', 'ori_input_mask',
                                 'aug_input_ids', 'aug_input_type_ids', 'aug_input_mask']
                self.tensors = [torch.tensor(data[c].apply(lambda x: ast.literal_eval(x)), dtype=torch.long)    \
                                                                                for c in input_columns]
                
            else:
                raise "d_type error. (d_type have to sup or unsup)"

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def get_sup(self, lines):
        raise NotImplementedError

    def get_unsup(self, lines):
        raise NotImplementedError

class IMDB_dataset(CsvDataset):
    labels = ('0', '1')
    def __init__(self, file, need_prepro, pipeline=[], max_len=128, mode='train', d_type='sup'):
        super().__init__(file, need_prepro, pipeline, max_len, mode, d_type)

    def get_sup(self, lines):
        for line in itertools.islice(lines, 0, None):
            yield line[7], line[6], []    # label, text_a, None
            # yield None, line[6], []

    def get_unsup(self, lines):
        for line in itertools.islice(lines, 0, None):
            yield (None, line[1], []), (None, line[2], [])  # ko, en

class matres_unsup_dataset(Dataset):
    labels = None
    def __init__(self, file, max_len):
        #dont have aug
        Dataset.__init__(self)
        self.cnt = 0
        # need preprocessing
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # unsupervised dataset
        data = {'ori':[], 'aug':[]}
        tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
        # for line in lines:
        #     breakpoint()
        #     ori = tokenizer(line.strip(),
        #         max_length=max_len,
        #         padding="max_length",
        #         truncation=True,)

        #     self.cnt += 1
        #     data['ori'].append(ori)    # drop label_id
        # ori_tensor = [torch.tensor(x, dtype=torch.long) for x in zip(*data['ori'])]
        ori_tokenized = tokenizer([(line.strip(),None ) for line in lines],
                max_length=max_len,
                padding="max_length",
                truncation=True,)
        ori_tensor = [torch.LongTensor(x) for x in [ori_tokenized['input_ids'],ori_tokenized['token_type_ids'],ori_tokenized['attention_mask']]]
        #just duplicate orig to be aug
        aug_tensor = ori_tensor
        self.tensors = ori_tensor + aug_tensor
        # already preprocessed
        
    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)
