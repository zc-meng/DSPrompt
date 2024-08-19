import torch
from torch import nn, optim
from torch.nn import functional as F
import pdb
import numpy as np
import random
 
 
class PassageAttention(nn.Module):
    def __init__(self,
                 passage_encoder,
                 num_class,
                 rel2id):
        """
        Args:
            passage_encoder: encoder for whole passage (bag of sentences)
            num_class: number of classes
        """
        super().__init__()
        self.passage_encoder = passage_encoder
        self.embed_dim = self.passage_encoder.hidden_size
        self.hidden_dim = 64
        self.num_class = num_class
        self.sigm = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        for rel, ids in rel2id.items():
            self.id2rel[ids] = rel

        self.fc1 = nn.Linear(self.embed_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.hidden_dim)
        self.get_mu = nn.Linear(self.embed_dim, self.hidden_dim)
        self.get_logvar = nn.Linear(self.embed_dim, self.hidden_dim)
        self.recon = nn.Linear(self.hidden_dim*3, self.embed_dim)
        self.cls = nn.Linear(self.embed_dim*3, self.embed_dim)
        self.fc = nn.Linear(self.embed_dim, self.num_class)
        self.dropout = nn.Dropout(0.1)
        ###
 
    def forward(self, token, mask, train=True):
        """
        Args:
            token: (nsum, L), index of tokens
            mask: (nsum, L), used for piece-wise CNN
            class_index: (bs)
        Return:
            logits, (B, N)
        """
        passage = self.passage_encoder(token, mask)  # (B, max_len, H)
 
        if train:
            rep_rel = passage[:, 160, :] # [REL] token
            rep_cls = passage[:, 0, :]  # [CLS] token

            ### Relation Abstraction Process
            mu = self.get_mu(rep_rel)
            logvar = self.get_logvar(rep_rel)
            var = torch.exp(0.5*logvar)
            z = torch.randn_like(var)
            ### Reparameterization to get generalized relation representation I
            I = mu + var * z # (batch_size, hidden_size)

            ### Context Injection Process
            e_h = passage[:, 129:160, :]
            e_h = torch.sum(e_h, dim=1)
            e_t = passage[:, 161:, :]
            e_t = torch.sum(e_t, dim=1)
            e_hh = self.fc1(e_h)
            e_tt = self.fc2(e_t)
            ### Inject the entity pair information into the generalized relation representation
            temp_recon = torch.cat((I, e_hh, e_tt), -1) # (batch_size, hidden_size*3)
            rep_recon = self.recon(temp_recon) # (batch_size, hidden_size)
 
            final_rep = torch.cat((rep_rel, e_h, e_t), -1)
            final_rep = self.cls(final_rep)
            rel_emb = self.dropout(final_rep)
            final_logits = self.fc(rel_emb) # (batch_size, num_class)
            rel_scores = self.sigm(final_logits)
 
        else:
            with torch.no_grad():
                rep_rel = passage[:, 160, :] 
                rep_cls = passage[:, 0, :]
 
                e_h = passage[:, 129:160, :]
                e_h = torch.sum(e_h, dim=1)
                e_t = passage[:, 161:, :]
                e_t = torch.sum(e_t, dim=1)
 
                mu = self.get_mu(rep_rel)
                logvar = self.get_logvar(rep_rel)
                var = torch.exp(0.5*logvar1)
                z = torch.randn_like(var1)
                I = mu + var * z
                e_hh = self.fc1(e_h)
                e_tt = self.fc2(e_t)
                temp_recon = torch.cat((I, e_hh, e_tt), -1)
                rep_recon = self.recon(temp_recon)
 
                final_rep = torch.cat((rep_rel, e_h, e_t), -1)
                final_rep = self.cls(final_rep)
 
                rel_emb = self.dropout(final_rep)
                final_logits = self.fc(rel_emb) # (B, num_class)
                rel_scores = self.sigm(final_logits)
 
        return rel_scores, mu, logvar, rep_rel, rep_cls, rep_recon