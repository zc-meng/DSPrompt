import logging
import json
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import random
 
 
class PassageEncoder(nn.Module):
    def __init__(self, pretrain_path, batch_size, blank_padding=True, mask_entity=False):
        super().__init__()
        self.blank_padding = blank_padding
        self.hidden_size = 768
        self.batch_size = batch_size
        self.mask_entity = mask_entity
        self.max_length = 128
 
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
 
    def forward(self, token, att_mask):
        hidden, _ = self.bert(token, attention_mask=att_mask)
        return hidden
 
    def tokenize(self, bag, data):
        max_len = 0
        indexed_tokens = []
        for it, sent_id in enumerate(bag):
            item = data[sent_id]
            if 'text' in item:
                sentence = item['text']
                is_token = False
            else:
                sentence = item['token']
                is_token = True
            pos_head = item['h']['pos']
            pos_tail = item['t']['pos']
 
            pos_min = pos_head
            pos_max = pos_tail
            if pos_head[0] > pos_tail[0]:
                pos_min = pos_tail
                pos_max = pos_head
                rev = True
            else:
                rev = False
 
            if not is_token:
                sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
                ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
                sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
                ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
                sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
            else:
                sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
                ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
                sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
                ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
                sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))
                sent_temp = " ".join(sentence)
    
            ent0 = ['[unused1]'] + ent0 + ['[unused2]'] if not rev else ['[unused3]'] + ent0 + ['[unused4]']
            ent1 = ['[unused3]'] + ent1 + ['[unused4]'] if not rev else ['[unused1]'] + ent1 + ['[unused2]']
 
            if it == 0:
                re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2
            else:
                re_tokens = sent0 + ent0 + sent1 + ent1 + sent2
            
            ###
            ent0_tokens = self.tokenizer.convert_tokens_to_ids(['[SEP]'] + ent0)
            length = len(ent0_tokens)
            if self.blank_padding:
                while len(ent0_tokens) < 32:
                    ent0_tokens.append(0)  # 0 is id for [PAD]
            ent0_tokens = ent0_tokens[:32]
            ent0_tokens.append(self.tokenizer.convert_tokens_to_ids('[unused10]'))
            ent0_tokens = torch.tensor(ent0_tokens).long().unsqueeze(0)  # (1, L)
            att_mask2_temp1 = torch.zeros(ent0_tokens.size()).long()  # (1, L)
            att_mask2_temp1[0 :length] = 1
            
            ent1_tokens = self.tokenizer.convert_tokens_to_ids(ent1)
            length = len(ent1_tokens)
            if self.blank_padding:
                while len(ent1_tokens) < 31:
                    ent1_tokens.append(0)  # 0 is id for [PAD]
            ent1_tokens = ent1_tokens[:31]
            ent1_tokens = torch.tensor(ent1_tokens).long().unsqueeze(0)  # (1, L)
            att_mask2_temp2 = torch.zeros(ent1_tokens.size()).long()  # (1, L)
            att_mask2_temp2[0 :length] = 1
            
            question_tokens = torch.cat((ent0_tokens, ent1_tokens), 1)
            att_mask3 = torch.cat((att_mask2_temp1, att_mask2_temp2), 1) # (1, L)
            ###
 
            curr_indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
            curr_len = len(curr_indexed_tokens)
            if max_len + curr_len <= self.max_length:
                indexed_tokens += curr_indexed_tokens
                max_len += curr_len
            else:
                if max_len == 0:
                    indexed_tokens = curr_indexed_tokens[:self.max_length]
                    max_len = len(indexed_tokens)
 
        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)
 
        # Attention mask
        att_mask1 = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask1[0, :max_len] = 1
        
        indexed_tokens = torch.cat((indexed_tokens, question_tokens), 1)
        att_mask = torch.cat((att_mask1, att_mask3), 1)
 
        return indexed_tokens, att_mask