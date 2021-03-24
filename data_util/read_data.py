DATA_DIR = '/home/xiongyi/dataxyz/data/counterfactually-augmented-data/sentiment/combined/paired'
IMDB_DIR = '/home/xiongyi/dataxyz/data/imdb/full'
import os
from .data_utils import InputExample, convert_examples_to_features, PairedBertDataset,SingleBertDataset
import pandas as pd
def read_imdb_cf_data(prefix = 'train',single = False, data_dir = DATA_DIR, max_len = 128, num = 100):
    if not single:
        label_2_id = {'Positive':1, 'Negative':0}
        with open (os.path.join(data_dir, '{}_paired.tsv'.format(prefix)), 'r') as f:
            lines = f.readlines()
        orig_sents, cf_sents, orig_labels, cf_labels = [],[],[],[]
        for i, line in enumerate(lines[1:]):
            label, sent = line.strip().split('\t')[:2]
            label = label_2_id[label]
            words = sent.strip().split(' ')
            words = words[:max_len]
            new_sent = ' '.join(words)
            if i % 2 == 0:
                orig_sents.append(new_sent)
                orig_labels.append(label)
            else:
                cf_sents.append(new_sent)
                cf_labels.append(label)
        ##only keeping those that changed during the first 100 words
        drop_ids = []
        for j in range(len(orig_sents)):
            if orig_sents[j] == cf_sents[j]:
                drop_ids.append(j)
        print( 'Dropped {} sents out of {}'.format(len(drop_ids), len(orig_sents)))

        new_orig_sents = [sent for jj, sent in enumerate(orig_sents) if jj not in drop_ids]
        new_orig_labels = [label for jj, label in enumerate(orig_labels) if jj not in drop_ids]
        new_cf_sents = [sent for jj, sent in enumerate(cf_sents) if jj not in drop_ids]
        new_cf_labels = [label for jj, label in enumerate(cf_labels) if jj not in drop_ids]

        return new_orig_sents,new_orig_labels, new_cf_sents,new_cf_labels
    else:
        with open (os.path.join(data_dir), 'r') as f:
            lines = f.readlines()
        sents = [l.strip().split('\t')[0] for l in lines]
        labels = [int(l.strip().split('\t')[-1]) for l in lines]
        return sents, labels


def read_imdb_contrast_data(prefix = 'dev', data_dir = '/home/xiongyi/dataxyz/data/contrast-sets/IMDb/data/', max_len = 128, num = 100):
    name_2_label = {'Negative':0, 'Positive':1}
    orig_file_name = '{}_original.tsv'.format(prefix)
    orig_df = pd.read_csv(os.path.join(data_dir, orig_file_name), sep='\t')
    orig_sents = list(orig_df['Text'])
    orig_labels = list(orig_df['Sentiment'])
    orig_labels = [name_2_label[l] for l in orig_labels]

    cf_file_name = '{}_contrast.tsv'.format(prefix)
    cf_df = pd.read_csv(os.path.join(data_dir, cf_file_name), sep='\t')
    cf_sents = list(cf_df['Text'])
    cf_labels = list(cf_df['Sentiment'])
    cf_labels = [name_2_label[l] for l in cf_labels]
    return orig_sents,orig_labels,cf_sents, cf_labels

def read_matres_data(prefix = 'train',single = False, data_dir = '/home/xiongyi/dataxyz/data/contrast-sets/', max_len = 128, num = 100):
    name_2_label = {'before':0, 'after':1}
    file_name = 'matres_{}.csv'.format(prefix)
    df = pd.read_csv(os.path.join(data_dir, file_name))
    df = df[df.decision != 'vague']
    df = df[df['new decision'] != 'vague']
    orig_sents = list(df['bodygraph'])
    orig_labels = list(df['decision'])
    orig_labels = [name_2_label[l.strip()] for l in orig_labels]
    cf_sents = list(df['modified bodygraph'])
    cf_labels = list(df['new decision'])
    cf_labels = [name_2_label[l.strip()] for l in cf_labels]
    return orig_sents,orig_labels,cf_sents, cf_labels


def read_mctaco_data(prefix = 'train', single = False,data_dir = '/home/xiongyi/dataxyz/data/boolq/', max_len = 128):
    name_2_label = {'no':0, 'yes':1}
    file_name = 'paired_mctaco_{}.txt'.format(prefix)
    with open(os.path.join(data_dir,file_name), 'r') as f:
        lines = f.readlines()
    orig_sents,orig_labels,cf_sents, cf_labels = [],[],[],[]
    for line in lines:
        orig_sent0, orig_sent1, orig_label, cf_sent0, cf_sent1, cf_label = line.strip().split('\t')
        orig_label = name_2_label[orig_label]
        cf_label = name_2_label[cf_label]
        orig_sents.append('\t\t'.join([orig_sent0,orig_sent1]))
        cf_sents.append('\t\t'.join([cf_sent0,cf_sent1]))
        orig_labels.append(orig_label)
        cf_labels.append(cf_label)
    return orig_sents,orig_labels,cf_sents, cf_labels



# def get_sup_dataset(tokenizer,orig_sents,orig_labels,cf_sents, cf_labels, batch_size = 8,max_len=128,text_b = False):
#     if not text_b:
#         orig_examples = [InputExample(guid= str(i), text_a = sent, label= orig_labels[i]) for i,sent in enumerate(orig_sents)]
#         cf_examples = [InputExample(guid= str(i), text_a = sent, label= cf_labels[i]) for i,sent in enumerate(cf_sents)]
#         orig_features = convert_examples_to_features(orig_examples, tokenizer, max_length=max_len, )
#         cf_features = convert_examples_to_features(cf_examples, tokenizer, max_length=max_len, )
#         dataset = PairedBertDataset(orig_features, cf_features)
#     else:
#         orig_examples = [InputExample(guid= str(i), text_a = sent.split('\t\t')[0],text_b=sent.split('\t\t')[1], label= orig_labels[i]) for i,sent in enumerate(orig_sents)]
#         cf_examples = [InputExample(guid= str(i), text_a = sent.split('\t\t')[0],text_b=sent.split('\t\t')[1], label= cf_labels[i]) for i,sent in enumerate(cf_sents)]
#         orig_features = convert_examples_to_features(orig_examples, tokenizer, max_length=max_len, )
#         cf_features = convert_examples_to_features(cf_examples, tokenizer, max_length=max_len, )
#         dataset = PairedBertDataset(orig_features, cf_features)
#     return dataset
    
    
# def get_eval_dataset(tokenizer,orig_sents,orig_labels,cf_sents, cf_labels, batch_size = 8,max_len=128,text_b = False):
#     if not text_b:
#         orig_examples = [InputExample(guid= str(i), text_a = sent, label= orig_labels[i]) for i,sent in enumerate(orig_sents)]
#         cf_examples = [InputExample(guid= str(i), text_a = sent, label= cf_labels[i]) for i,sent in enumerate(cf_sents)]
#         orig_features = convert_examples_to_features(orig_examples, tokenizer, max_length=max_len, )
#         cf_features = convert_examples_to_features(cf_examples, tokenizer, max_length=max_len, )
#         orig_eval_dataset = SingleBertDataset(orig_features)
#         cf_eval_dataset = SingleBertDataset(cf_features)

#     return orig_eval_dataset,cf_eval_dataset
    
    