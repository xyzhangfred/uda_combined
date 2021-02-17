import logging
import os,sys
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator
from dataclasses import dataclass
import itertools
import ast

# sys.path.append("../..") # Adds higher directory to python modules path.

@dataclass
class InputExample:
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label = None, label_classes = None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label


def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer,
    max_length: Optional[int] = None,
    label_list=None,
    # output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        return int(example.label)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    return features

class SingleBertDataset(Dataset):

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]




class PairedBertDataset(Dataset):

    def __init__(self, orig_features, cf_features):
        self.orig_features = orig_features
        self.cf_features = cf_features
        assert len(self.orig_features) == len(self.cf_features)

    def __len__(self):
        return len(self.orig_features)

    def __getitem__(self, i) -> InputFeatures:
        return (self.orig_features[i], self.cf_features[i])

def paired_data_collator(paired_features):
    """
    Very simple data collator that:
    - simply collates batches of dict-like objects
    - Performs special handling for potential keys named:
        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object
    - does not do any additional preprocessing

    i.e., Property names of the input object will be used as corresponding inputs to the model.
    See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    paired_features = [(vars(f_0),vars(f_1))  for (f_0, f_1) in paired_features]

    first = paired_features[0][0]
    batch = {'orig':{}, 'cf':{}}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    batch['orig']["labels"] = torch.tensor([f[0]["label"] for f in paired_features], dtype=int)
    batch['cf']["labels"] = torch.tensor([f[0]["label"] for f in paired_features], dtype=int)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch['orig'][k] = torch.stack([f[0][k] for f in paired_features])
                batch['cf'][k] = torch.stack([f[0][k] for f in paired_features])
            else:
                batch['orig'][k] = torch.stack([torch.LongTensor(f[0][k]) for f in paired_features])
                batch['cf'][k] = torch.stack([torch.LongTensor(f[0][k]) for f in paired_features])

    return batch



def default_data_collator(features):
    """
    Very simple data collator that:
    - simply collates batches of dict-like objects
    - Performs special handling for potential keys named:
        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object
    - does not do any additional preprocessing

    i.e., Property names of the input object will be used as corresponding inputs to the model.
    See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch

def get_paired_dataloader(tokenizer,orig_sents,orig_labels,cf_sents, cf_labels, batch_size = 8,max_len=128):
    orig_examples = [InputExample(guid= str(i), text_a = sent, label= orig_labels[i]) for i,sent in enumerate(orig_sents)]
    cf_examples = [InputExample(guid= str(i), text_a = sent, label= cf_labels[i]) for i,sent in enumerate(cf_sents)]

    orig_features = convert_examples_to_features(orig_examples, tokenizer, max_length=max_len, )
    cf_features = convert_examples_to_features(cf_examples, tokenizer, max_length=max_len, )
    dataset = PairedBertDataset(orig_features, cf_features)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=paired_data_collator, shuffle= True)

    return dataloader


def get_single_dataloader(tokenizer,sents,labels, batch_size = 8,max_len=128,shuffle = False):
    examples = [InputExample(guid= str(i), text_a = sent, label= labels[i]) for i,sent in enumerate(sents)]

    features = convert_examples_to_features(examples, tokenizer, max_length=max_len, )
    dataset = SingleBertDataset(features)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=default_data_collator, shuffle= shuffle)

    return dataloader


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

                    self.tensors = [torch.Tensor(x, dtype=torch.long) for x in zip(*data)]
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
                    ori_tensor = [torch.Tensor(x, dtype=torch.long) for x in zip(*data['ori'])]
                    aug_tensor = [torch.Tensor(x, dtype=torch.long) for x in zip(*data['aug'])]
                    self.tensors = ori_tensor + aug_tensor
        # already preprocessed
        else:
            f = open(file, 'r', encoding='utf-8')
            data = pd.read_csv(f, sep='\t')

            # supervised dataset
            if d_type == 'sup':
                # input_ids, segment_ids(input_type_ids), input_mask, input_label
                input_columns = ['input_ids', 'input_type_ids', 'input_mask', 'label_ids']
                self.tensors = [torch.Tensor(data[c].apply(lambda x: ast.literal_eval(x)), dtype=torch.long)    \
                                                                                for c in input_columns[:-1]]
                self.tensors.append(torch.Tensor(data[input_columns[-1]], dtype=torch.long))
                
            # unsupervised dataset
            elif d_type == 'unsup':
                input_columns = ['ori_input_ids', 'ori_input_type_ids', 'ori_input_mask',
                                 'aug_input_ids', 'aug_input_type_ids', 'aug_input_mask']
                self.tensors = [torch.Tensor(data[c].apply(lambda x: ast.literal_eval(x)), dtype=torch.long)    \
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


class IMDB(CsvDataset):
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


