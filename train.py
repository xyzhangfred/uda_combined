# Copyright 2019 SanghunYun, Korea University.
# (Strongly inspired by Dong-Hyun Lee, Kakao Brain)
# 
# Except load and save function, the whole codes of file has been modified and added by
# SanghunYun, Korea University for UDA.
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
import json
from copy import deepcopy
from typing import NamedTuple
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
# from utils import checkpoint
# from utils.logger import Logger
from tensorboardX import SummaryWriter
# from utils.utils import output_logging


class Trainer(object):
    """Training Helper class"""
    def __init__(self, cfg, model, data_iter, optimizer, device,results_dir):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        # self.optimizer = torch.optim.AdamW(model.projector.parameters(),lr=cfg.lr)
        # self.optimizer_2 = torch.optim.AdamW(list(model.bert.parameters())+list(model.classifier.parameters()),lr=cfg.lr)

        self.device = device
        self.results_dir = results_dir
        # data iter
        if len(data_iter) == 1:
            self.sup_iter = data_iter[0]
        elif len(data_iter) == 2:
            self.sup_iter = self.repeat_dataloader(data_iter[0])
            self.eval_iter = data_iter[1]
        elif len(data_iter) == 3:
            self.sup_iter = self.repeat_dataloader(data_iter[0])
            self.unsup_iter = self.repeat_dataloader(data_iter[1])
            self.eval_iter = data_iter[2]
        elif len(data_iter) == 4:
            self.sup_iter = self.repeat_dataloader(data_iter[0])
            self.unsup_iter = self.repeat_dataloader(data_iter[1])
            self.orig_eval_iter = data_iter[2]
            self.cf_eval_iter = data_iter[3]

    def train(self, get_loss, get_acc, model_file, pretrain_file):
        """ train uda"""
        # tensorboardX logging
        # if self.cfg.self.results_dir:
        #     logger = SummaryWriter(log_dir=os.path.join(self.cfg.self.results_dir, 'logs'))
        logger = SummaryWriter(log_dir=os.path.join(self.results_dir, 'logs'))
        acc_file = os.path.join(self.results_dir, 'acc.txt')
        with open (acc_file, 'a') as f:
            f.write('Step, Acc_orig, Acc_cf\n')
        loss_file = os.path.join(self.results_dir, 'eval_loss.txt')
        with open (loss_file, 'a') as f:
            f.write('Step, Proj_loss, Sup_loss\n')
        self.model.bert.train()
        self.model.bert.to(self.device)
        self.model.projector.train()
        self.model.projector.to(self.device)
        self.model.classifier.train()
        self.model.classifier.to(self.device)

        global_step = 0
        loss_sum = 0.
        max_acc_cf = [0., 0]   # acc, step

        # Progress bar is set by unsup or sup data
        # uda_mode == True --> sup_iter is repeated
        # uda_mode == False --> sup_iter is not repeated
        iter_bar = tqdm(self.unsup_iter, total=self.cfg.total_steps) if self.cfg.uda_mode \
              else tqdm(self.sup_iter, total=self.cfg.total_steps)
        for i, batch in enumerate(iter_bar):
            # Device assignment
            if self.cfg.uda_mode:
                sup_batch_full = next(self.sup_iter)
                sup_batch_orig = sup_batch_full['orig']
                sup_batch_cf = sup_batch_full['cf']
                sup_batch_orig = {t: sup_batch_orig[t].to(self.device) for t in sup_batch_orig}
                sup_batch_cf = {t: sup_batch_cf[t].to(self.device) for t in sup_batch_cf}
                sup_batch = [sup_batch_orig, sup_batch_cf]
                unsup_batch = [t.to(self.device) for t in batch]
            
            else:
                sup_batch = [t.to(self.device) for t in batch]
                unsup_batch = None

            # update
            self.optimizer.zero_grad()
            final_loss, sup_loss, unsup_loss,ruda_loss,proj_loss = get_loss(self.model, sup_batch, unsup_batch, global_step)
            # final_loss_x_ruda = sup_loss+unsup_loss+proj_loss
            # final_loss_x_ruda.backward(retain_graph=True)
            final_loss.backward()
            self.optimizer.step()
            # self.optimizer_2.zero_grad()
            # final_loss.backward()
            # self.optimizer_2.step()

            # print loss
            global_step += 1
            loss_sum += final_loss.item()
            if self.cfg.uda_mode:
                iter_bar.set_description('final=%5.3f ruda=%5.3f proj=%5.3f sup=%5.3f uda=%5.3f'\
                        % (final_loss.item(), ruda_loss.item(),proj_loss.item(), sup_loss.item(), unsup_loss.item()))
            else:
                iter_bar.set_description('loss=%5.3f' % (final_loss.item()))

            # logging            
            if self.cfg.uda_mode:
                logger.add_scalars('data/scalar_group',
                                    {'final_loss': final_loss.item(),
                                     'sup_loss': sup_loss.item(),
                                     'unsup_loss': unsup_loss.item(),
                                     'ruda_loss': ruda_loss.item(),
                                     'proj_loss': proj_loss.item()
                                    }, global_step)
            else:
                logger.add_scalars('data/scalar_group',
                                    {'sup_loss': final_loss.item()}, global_step)

            if global_step % self.cfg.save_steps == 0:
                self.save(global_step)

            if get_acc and global_step % self.cfg.check_steps == 0 or global_step == 1:
                eval_losses = self.eval_loss(self.model)
                results_orig = self.eval(get_acc, None, self.model, 'orig')
                total_accuracy_orig = torch.cat(results_orig).mean().item()
                logger.add_scalars('data/scalar_group', {'eval_acc_orig' : total_accuracy_orig}, global_step)
                results_cf = self.eval(get_acc, None, self.model, 'cf')
                total_accuracy_cf = torch.cat(results_cf).mean().item()
                logger.add_scalars('data/scalar_group', {'eval_acc_cf' : total_accuracy_cf}, global_step)
                if max_acc_cf[0] < total_accuracy_cf:
                    self.save(global_step)
                    max_acc_cf = total_accuracy_cf, global_step
                with open (acc_file, 'a') as f:
                    f.write('{}, {:.3f} , {:.3f}\n'.format(global_step, total_accuracy_orig,total_accuracy_cf))
                with open (loss_file, 'a') as f:
                    f.write('{}, {:.3f} , {:.3f}\n'.format(global_step, eval_losses['proj'],eval_losses['sup']))
                
                print('Accuracy Orig: %5.3f' % total_accuracy_orig)
                print('Accuracy CF: %5.3f' % total_accuracy_cf)
                print('Max Accuracy CF: %5.3f Max global_steps : %d Cur global_steps : %d' %(max_acc_cf[0], max_acc_cf[1], global_step), end='\n\n')

            if self.cfg.total_steps and self.cfg.total_steps < global_step:
                # print('The total steps have been reached')
                # print('Average Loss %5.3f' % (loss_sum/(i+1)))
                # if get_acc:
                #     results = self.eval(get_acc, None, self.model)
                #     total_accuracy = torch.cat(results).mean().item()
                #     logger.add_scalars('data/scalar_group', {'eval_acc' : total_accuracy}, global_step)
                #     if max_acc_cf[0] < total_accuracy:
                #         max_acc_cf = total_accuracy, global_step                
                #     print('Accuracy :', total_accuracy)
                #     print('Max Accuracy : %5.3f Max global_steps : %d Cur global_steps : %d' %(max_acc_cf[0], max_acc_cf[1], global_step), end='\n\n')
                # self.save(global_step)
                return
        return global_step

    def eval(self, evaluate, model_file, model, prefix = 'orig'):
        """ evaluation function """
        
        results = []
        if prefix == 'orig':
            iter_bar = tqdm(deepcopy(self.orig_eval_iter))
        elif prefix == 'cf':
            iter_bar = tqdm(deepcopy(self.cf_eval_iter))
        else:
            print ('Error')
        for batch in iter_bar:
            batch = {t:batch[t].to(self.device) for t in batch}

            with torch.no_grad():
                accuracy, result = evaluate(model, batch)
            results.append(result)

            iter_bar.set_description('Eval Acc {} ={:5.3f}'.format(prefix,accuracy))
        return results
            
    def eval_loss(self, model):
        sup_criterion = nn.CrossEntropyLoss(reduction='none')
        unsup_criterion = nn.KLDivLoss(reduction='none')
        sup_proj_criterion = nn.MSELoss(reduction='mean')

        all_eval_losses = {'sup':[], 'proj':[]}
        orig_iter_bar = tqdm(deepcopy(self.orig_eval_iter))
        cf_iter_bar = tqdm(deepcopy(self.cf_eval_iter))
        for orig_batch,cf_batch in zip(orig_iter_bar,cf_iter_bar):
            orig_batch = {t: orig_batch[t].to(self.device) for t in orig_batch}
            cf_batch = {t: cf_batch[t].to(self.device) for t in cf_batch}
            # Device assignment
            # sup loss
            hid_orig = model.bert(input_ids=orig_batch['input_ids'],attention_mask = orig_batch['attention_mask'],token_type_ids=orig_batch['token_type_ids'])[1]
            hid_cf = model.bert(input_ids=cf_batch['input_ids'],attention_mask = cf_batch['attention_mask'],token_type_ids=cf_batch['token_type_ids'])[1]
            logits_orig = model.classifier(hid_orig)      
            logits_cf = model.classifier(hid_cf)      
            try:
                sup_loss = sup_criterion(logits_orig, orig_batch['labels'])+sup_criterion(logits_cf, cf_batch['labels'])  # shape : train_batch_size
            except:
                breakpoint()
            #what if we learn the residual
            # hid_sup_proj = model.projector(hid_orig_sup) + hid_orig_sup
            hid_cf_proj = model.projector(hid_orig) 
            proj_loss = sup_proj_criterion(hid_cf_proj,hid_cf).mean()
            sup_loss = torch.mean(sup_loss)
            all_eval_losses['proj'].append(proj_loss.item())
            all_eval_losses['sup'].append(sup_loss.item())
            # unsup loss
        eval_losses = {t:np.mean(v) for t,v in all_eval_losses.items()} 
        return eval_losses

    def load(self, model_file, pretrain_file):
        """ between model_file and pretrain_file, only one model will be loaded """
        if model_file:
            print('Loading the model from', model_file)
            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load(model_file))
            else:
                self.model.load_state_dict(torch.load(model_file, map_location='cpu'))

        elif pretrain_file:
            print('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.ckpt'):  # checkpoint file in tensorflow
                checkpoint.load_model(self.model.transformer, pretrain_file)
            elif pretrain_file.endswith('.pt'):  # pretrain model file in pytorch
                self.model.transformer.load_state_dict(
                    {key[12:]: value
                        for key, value in torch.load(pretrain_file).items()
                        if key.startswith('transformer')}
                )   # load only transformer parts
    
    def save(self, i):
        """ save model """
        if not os.path.isdir(os.path.join(self.results_dir, 'save')):
            os.makedirs(os.path.join(self.results_dir, 'save'))
        torch.save(self.model.bert.state_dict(),
                        os.path.join(self.results_dir, 'save', 'bert_steps_'+str(i)+'.pt'))
        torch.save(self.model.projector.state_dict(),
                        os.path.join(self.results_dir, 'save', 'proj_steps_'+str(i)+'.pt'))
        torch.save(self.model.classifier.state_dict(),
                        os.path.join(self.results_dir, 'save', 'cls_steps_'+str(i)+'.pt'))    

    def repeat_dataloader(self, iterable):
        """ repeat dataloader """
        while True:
            for x in iterable:
                yield x
