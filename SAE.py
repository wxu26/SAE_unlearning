"""
Functions and global variables from train_SAE.ipynb

main variables/objects and functions to be imported from here:
    config_default
    TransformerWithSAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from model import config_default, model_data, TransformerModel

config_default['n_feature'] = 2048
config_default['lambda'] = 2 # regularization term in SAE loss
config_default['layer_for_SAE'] = 6 # SAE after this layer

DEFAULT_MODEL = "model.10000.pth"

class TransformerWithSAE(TransformerModel):
    def __init__(self, config=config_default):
        super().__init__()
        # freeze old parameters
        for p in self.parameters():
            p.requires_grad = False
        # switch to eval mode (this is mainly for dropout and does not affect SAE)
        self.eval()
        # SAE
        self.W_enc = nn.Parameter(torch.randn(config['n_feature'], config['n_model'])/np.sqrt(config['n_model']))
        self.W_dec = nn.Parameter(self.W_enc.data)
        self.b_enc = nn.Parameter(torch.zeros(config['n_feature']))
        self.b_dec = nn.Parameter(torch.zeros(config['n_model']))
        # other SAE parameters
        self.lam = config['lambda']
        self.layer_for_SAE = config['layer_for_SAE']
        self.scale_factor = 1 # should be updated by calling self.update_scale_factor()
    def forward(self, t, y=None, SAE_loss=False):
        """
        SAE loss:
        determines whether to output feature amp + SAE loss
        or logits + model loss (corss entropy with y)
        (default to False so that model.generate() works properly)
        """
        # run until SAE
        x = self.embed(t)
        x_in = self.layers[:self.layer_for_SAE](x) * self.scale_factor
        # SAE
        features = torch.einsum('...i,ji->...j',x_in,self.W_enc) + self.b_enc
        features = F.relu(features)
        x_out = torch.einsum('...i,ij->...j',features,self.W_dec) + self.b_dec
        # SAE loss
        if SAE_loss:
            # loss function from here:
            # https://transformer-circuits.pub/2024/april-update/index.html#training-saes
            L2_loss = torch.sum((x_out-x_in)**2, axis=-1)
            reuglarization = self.lam*torch.sum(features*torch.norm(self.W_dec,dim=-1), axis=-1) # feature is non-negative already
            loss = torch.mean(L2_loss + reuglarization)
            return features, loss
        # after SAE
        x = self.layers[self.layer_for_SAE:](x_out*(1./self.scale_factor))
        x = self.final_norm(x)
        logits = self.final_lin(x)
        if y is None:
            return logits, None
        logits = logits.view(-1,self.n_token)
        y = y.view(-1)
        loss = F.cross_entropy(logits, y)
        return logits, loss
    @torch.no_grad()
    def update_scale_factor(self, model_data=model_data, indices=range(256*1,256*(1+256*14),256*14)):
        """
        draw a sample from model data and use it to compute scale factor
        for reproducbility, the indices are fixed to 256 contexts (256*256 activations)
        across the training set
        """
        d = model_data.x_train
        t = torch.stack([d[i:i+model_data.n_context] for i in indices])
        x = self.embed(t)
        x = self.layers[:self.layer_for_SAE](x)
        # scale so that x^2 = n_model - so element-wise x_i^2 averages to 1
        self.scale_factor = torch.mean(x**2).item()**-0.5
        # print(f"scale action by {self.scale_factor:.3e}")
        return