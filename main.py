# Copyright 2019 SanghunYun, Korea University.
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

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# import models
import train
from data_util.load_data import load_data
# from utils.utils import get_device, _get_device, torch_device_one
from utils import optim, configuration
from transformers import BertModel
from model import BERTProjector,BERTResProjector
# TSA
def get_tsa_thresh(schedule, global_step, num_train_steps, start, end,device):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    output = threshold * (end - start) + start
    return output.to(device)


def main(cfg, model_cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument('--p', type=float, help='coeff for projection loss')
    parser.add_argument('--r', type=float, help='coeff for ruda loss')
    parser.add_argument('--u', type=float, help='coeff for uda loss')
    parser.add_argument('--hidden_dim', type=int, default = 768, help='hidden_dim for projector')
    parser.add_argument('--layer_num', type=int, default = 4, help='layer_num for projector')
    parser.add_argument('--results_dir', type=str, default = None, help='result file name.')
    parser.add_argument('--sup_data_dir', type=str, default = None, help='sup data dir.')
    parser.add_argument('--eval_data_dir', type=str, default = None, help='eval_data_dir')
    parser.add_argument('--data_type', type=str, default = None, help='imdb cf or imdb contrast',)
    args = parser.parse_args()
    
    # Load Configuration
    if args.config is not None:
        cfg = args.config        
   
    cfg = json.load(open(cfg,'r'))
    if args.sup_data_dir is not None:
        cfg['sup_data_dir'] = args.sup_data_dir        
    if args.eval_data_dir is not None:          
        cfg['eval_data_dir'] = args.eval_data_dir                  
    if args.data_type is not None:
        cfg['data_type'] = args.data_type     
    if args.u is not None:
        cfg['uda_coeff'] = args.u            
    cfg = configuration.params.from_dict(cfg) 

    model_cfg = configuration.model.from_json(model_cfg)        # BERT_cfg
    proj_coeff = args.p
    ruda_coeff = args.r

    if args.results_dir is None:
        results_dir = cfg.results_dir +'/p_{}_r_{}'.format(proj_coeff,ruda_coeff)
    else:
        results_dir = args.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # SEED = 1
    # set_seeds(42)
    # Load Data & Create Criterion
    data = load_data(cfg)
    if cfg.uda_mode:
        unsup_criterion = nn.KLDivLoss(reduction='none')
        orig_eval_iter,cf_eval_iter = data.eval_data_iter()
        data_iter = [data.sup_data_iter(), data.unsup_data_iter(), orig_eval_iter,cf_eval_iter]  # train_eval
    else:
        data_iter = [data.sup_data_iter(),data.eval_data_iter()]
    sup_criterion = nn.CrossEntropyLoss(reduction='none')
    sup_proj_criterion = nn.MSELoss(reduction='mean')
    # Load Model
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    #hid_dim = 768 or 300
    model = BERTResProjector(bert_model, input_dim=768,hidden_dim=args.hidden_dim, output_dim=768, block_num=args.layer_num)
    device = torch.device('cuda:0')
    optimizer = torch.optim.AdamW(list(model.bert.parameters())+list(model.projector.parameters())+list(model.classifier.parameters()),lr=cfg.lr)
    # Create trainer
    torch.autograd.set_detect_anomaly(True)
    trainer = train.Trainer(cfg, model, data_iter, optimizer, device, results_dir)

    # Training
    def get_loss(model, sup_batch, unsup_batch, global_step):

        # logits -> prob(softmax) -> log_prob(log_softmax)
        # batch
        sup_batch_orig, sup_batch_cf = sup_batch
        
        if unsup_batch:
            ori_input_ids, ori_segment_ids, ori_input_mask, \
            aug_input_ids, aug_segment_ids, aug_input_mask  = unsup_batch
            # input_ids = torch.cat((input_ids, aug_input_ids), dim=0)
            # segment_ids = torch.cat((segment_ids, aug_segment_ids), dim=0)
            # input_mask = torch.cat((input_mask, aug_input_mask), dim=0)
            
        # sup loss
        sup_size = len(sup_batch_orig['input_ids']) 
        unsup_size = unsup_batch[0].shape[0]
        hid_orig_sup = model.bert(input_ids=sup_batch_orig['input_ids'],attention_mask = sup_batch_orig['attention_mask'],\
            token_type_ids=sup_batch_orig['token_type_ids'])[1]
        hid_cf_sup = model.bert(input_ids=sup_batch_cf['input_ids'],attention_mask = sup_batch_cf['attention_mask'],
        token_type_ids=sup_batch_cf['token_type_ids'])[1]
        logits_orig_sup = model.classifier(hid_orig_sup)      
        logits_cf_sup = model.classifier(hid_cf_sup)      
        try:
            sup_loss = sup_criterion(logits_orig_sup, sup_batch_orig['labels'])+sup_criterion(logits_cf_sup, sup_batch_cf['labels'])  # shape : train_batch_size
        except:
            breakpoint()
        #what if we learn the residual
        # hid_sup_proj = model.projector(hid_orig_sup) + hid_orig_sup
        hid_sup_proj = model.projector(hid_orig_sup) 
        proj_loss = sup_proj_criterion(hid_sup_proj,hid_cf_sup).mean()
        if cfg.tsa:
            tsa_thresh = get_tsa_thresh(cfg.tsa, global_step, cfg.total_steps, start=1./logits_orig_sup.shape[-1], end=1,device=device)
            larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh   # prob = exp(log_prob), prob > tsa_threshold
            # larger_than_threshold = torch.sum(  F.softmax(pred[:sup_size]) * torch.eye(num_labels)[sup_label_ids]  , dim=-1) > tsa_threshold
            loss_mask = torch.ones(sup_size, dtype=torch.float32, device=device) * (1 - larger_than_threshold.type(torch.float32))
            sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch.tensor(1.).to(device))
        else:
            sup_loss = torch.mean(sup_loss)

        # unsup loss
        if unsup_batch:
            # ori
            with torch.no_grad():
                orig_hid = model.bert(input_ids = ori_input_ids, attention_mask = ori_input_mask,token_type_ids=ori_segment_ids)[1]
                ori_logits = model.classifier(orig_hid)
                ori_prob   = F.softmax(ori_logits, dim=-1)    # KLdiv target

                # confidence-based masking
                if cfg.uda_confidence_thresh != -1:
                    unsup_loss_mask = torch.max(ori_prob, dim=-1)[0] > cfg.uda_confidence_thresh
                    unsup_loss_mask = unsup_loss_mask.type(torch.float32)
                else:
                    unsup_loss_mask = torch.ones(unsup_size, dtype=torch.float32)
                unsup_loss_mask = unsup_loss_mask.to(device)
                    
            # aug
            # softmax temperature controlling
            uda_softmax_temp = cfg.uda_softmax_temp if cfg.uda_softmax_temp > 0 else 1.
            aug_logits = model.classifier(model.bert(input_ids = aug_input_ids, attention_mask = aug_input_mask,token_type_ids=aug_segment_ids)[1])
            aug_log_prob = F.log_softmax(aug_logits / uda_softmax_temp, dim=-1)

            #residual connection?
            # proj_unsup_hid = model.projector(orig_hid) + orig_hid
            proj_unsup_hid = model.projector(orig_hid)
            proj_unsup_logits = model.classifier(proj_unsup_hid)
            proj_unsup_log_prob = F.log_softmax(proj_unsup_logits / uda_softmax_temp, dim=-1)

            # KLdiv loss
            """
                nn.KLDivLoss (kl_div)
                input : log_prob (log_softmax)
                target : prob    (softmax)
                https://pytorch.org/docs/stable/nn.html

                unsup_loss is divied by number of unsup_loss_mask
                it is different from the google UDA official
                The official unsup_loss is divided by total
                https://github.com/google-research/uda/blob/master/text/uda.py#L175
            """
            unsup_loss = torch.sum(unsup_criterion(aug_log_prob, ori_prob), dim=-1)
            unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1), torch.tensor(1.).to(device))
            
            ruda_loss = torch.sum(unsup_criterion(proj_unsup_log_prob, ori_prob), dim=-1)
            ruda_loss = torch.sum(ruda_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(ruda_loss, dim=-1), torch.tensor(1.).to(device))
            ruda_loss = - ruda_loss
            # final_loss = sup_loss + cfg.uda_coeff*unsup_loss +cfg.ruda_coeff*ruda_loss + cfg.proj_coeff * proj_loss
            final_loss = sup_loss + cfg.uda_coeff*unsup_loss +ruda_coeff*ruda_loss + proj_coeff * proj_loss

            return final_loss, sup_loss, unsup_loss,ruda_loss,proj_loss
        return sup_loss, None, None

    # evaluation
    def get_acc(model, batch):
        # input_ids, segment_ids, input_mask, label_id, sentence = batch
        
        logits =model.classifier(model.bert(batch['input_ids'], batch['attention_mask'])[1])
        _, label_pred = logits.max(1)

        result = (label_pred == batch['labels']).float()
        accuracy = result.mean()
        # output_dump.logs(sentence, label_pred, label_id)    # output dump

        return accuracy, result

    if cfg.mode == 'train':
        trainer.train(get_loss, None, cfg.model_file, cfg.pretrain_file)

    if cfg.mode == 'train_eval' or cfg.mode == 'train_eval_cf':
        trainer.train(get_loss, get_acc, cfg.model_file, cfg.pretrain_file)


    if cfg.mode == 'eval':
        results = trainer.eval(get_acc, cfg.model_file, None)
        total_accuracy = torch.cat(results).mean().item()
        print('Accuracy :' , total_accuracy)


if __name__ == '__main__':
    main('config/uda.json', 'config/bert_base.json')
    # fire.Fire(main)
    # for rep in range(5):
        # for p in [0,1,2,5]:
            # for r in [0,1,2,5]:
                # results_dir = 'feb12cf_200_p_{}_r_{}_rep_{}'.format(p,r,rep)
