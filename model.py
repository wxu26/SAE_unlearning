"""
Functions and global variables from train_model.ipynb

main variables/objects and functions to be imported from here:
    config_default
    model_data
    TransformerModel
    generate_from_model()
    validation()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
default config for model + training
(the training parameters are not used for model initialization)
"""
config_default = {
    'n_layer':6,
    'n_model':256,
    'n_ff':1024,
    'n_head':8,
    'head_size':32, # prefer head_size * n_head = n_model; also needs to be even for RoPE
    'n_context':256,
    'n_token':66, # come from data
    'RoPE':True, # fix this to true as other embeddings have not been implemented yet
    'dropout':0.2, # higher dropout
    'n_batch':64,
    'lr':3e-4, # lower training rate
    'weight_decay':0.1,
}

class RoPEHead(nn.Module):
    """
    attention head using RoPE position embedding
    """
    # I need to turn k and q by some theta value that depends on the location in the context.
    # so this matrix can totally be precomputed... it's just a head_size * head_size thing.
    # or I could even say that I make a (head_size/2,2,2) tensor and reshape k, q to perform the rotation.
    def __init__(self, config=config_default):
        super().__init__()
        self.n_model = config['n_model']
        self.n_head = config['n_head']
        self.head_size = config['head_size']
        self.n_context = config['n_context']
        self.K = nn.Linear(self.n_model, self.head_size ,bias=False)
        self.Q = nn.Linear(self.n_model, self.head_size ,bias=False)
        self.V = nn.Linear(self.n_model, self.head_size ,bias=False)
        self.register_buffer('mask', 
          torch.triu(torch.full((self.n_context,self.n_context), float('-inf')), diagonal=1))
          # used for masking attention weights
        rotation_matrix = torch.zeros((self.head_size//2, 2, 2))
        exponents = torch.arange(self.head_size//2)/(self.head_size//2)
        thetas = 10000**(-exponents)
        self.register_buffer('cos', torch.cos(thetas))
        self.register_buffer('sin', torch.sin(thetas))
    def forward(self, x):
        # x: (..., n_context, n_model)
        k = self.K(x) #  (..., n_context, head_size)
        q = self.K(x)
        v = self.K(x)
        
        # rotate: spliting the tensor is faster than matrix multiplication
        k1 = k[...,::2]
        k2 = k[...,1::2]
        k[...,::2] = k1*self.cos + k2*self.sin
        k[...,1::2] = -k1*self.sin + k2*self.cos
        q1 = q[...,::2]
        q2 = q[...,1::2]
        q[...,::2] = q1*self.cos + q2*self.sin
        q[...,1::2] = -q1*self.sin + q2*self.cos
        
        w = torch.einsum('...ik, ...jk -> ...ij', k, q) * self.head_size**-0.5 # (..., n_context, n_context)
        # mask upper half with -inf (will become zero in softmax)
        w = w+self.mask
        w = F.softmax(w, dim=-1)
        out = torch.einsum('...ij, ...jk -> ...ik', w, v) #  (..., n_context, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, config=config_default):
        super().__init__()
        self.n_head = config['n_head']
        if config['RoPE']:
            self.heads = nn.ModuleList([RoPEHead(config) for i in range(self.n_head)])
        else:
            self.heads = nn.ModuleList([Head(config) for i in range(self.n_head)])
        self.linear = nn.Linear(self.n_head*config['head_size'], config['n_model'])
        self.dropout = nn.Dropout(config['dropout'])
    def forward(self, x):
        outs = [h(x) for h in self.heads]
        out = torch.cat(outs, dim=-1)
        return self.dropout(self.linear(out))

class FeedForward(nn.Module):
    """
    standard GELU feed forward
    use no bias (preferred by recent models)
    """
    def __init__(self, config=config_default):
        super().__init__()
        n_model, n_ff = config['n_model'], config['n_ff']
        self.FF = nn.Sequential(
            nn.Linear(n_model, n_ff, bias=False),
            nn.GELU(),
            nn.Linear(n_ff, n_model, bias=False),
            nn.Dropout(config['dropout'])
        )
    def forward(self, x):
        return self.FF(x)

class TransformerLayer(nn.Module):
    """
    one layer in the transformer
    """
    def __init__(self, config=config_default):
        super().__init__()
        self.MHA = MultiHeadAttention(config)
        self.FF = FeedForward(config)
        n_model = config['n_model']
        self.MHA_norm = nn.LayerNorm(n_model) # wanted to use MSE but it's not on the pytorch version in my env...
        self.FF_norm = nn.LayerNorm(n_model)
    def forward(self, x):
        x = x + self.MHA(self.MHA_norm(x))
        x = x + self.FF(self.FF_norm(x))
        return x

class TransformerModel(nn.Module):
    """
    the whole transformer
    """
    def __init__(self, config=config_default):
        super().__init__()
        n_layer = config['n_layer']
        n_model = config['n_model']
        n_token = config['n_token']
        self.n_token = n_token
        self.layers = nn.Sequential(*[TransformerLayer(config) for i in range(n_layer)])
        self.embed = nn.Embedding(n_token, n_model)
        self.final_norm = nn.LayerNorm(n_model)
        self.final_lin = nn.Linear(n_model, n_token)
        # scratch arrays for recording training
        self.current_epoch = 0
        self.losses = []
        self.diag_epochs = []
        self.diag_train_loss = []
        self.diag_val_loss = []
        # also keep a copy of config
        self.config = config
    def forward(self, t, y=None):
        x = self.embed(t)
        x = self.layers(x)
        x = self.final_norm(x)
        logits = self.final_lin(x)
        if y is None:
            return logits, None # update from notebook version: added None as second output here
        logits = logits.view(-1,self.n_token)
        y = y.view(-1)
        loss = F.cross_entropy(logits, y)
        return logits, loss
    def generate(self, t):
        # generate one new token (still encoded) given t
        logits = self.forward(t)[0][..., -1, :] # just the last row, using full context
        probs = F.softmax(logits, dim=-1)
        y = torch.multinomial(probs, num_samples=1)[...,0]
        return y

class ModelData():
    def __init__(self, train="train.csv", val="validation.csv", test="test.csv", config=config_default):
        """
        inputs: filenames
        """
        with open(train) as f:
            self.train = f.read()
            self.n_train = len(self.train)
        with open(val) as f:
            self.val = f.read()
            self.n_val = len(self.val)
        with open(test) as f:
            self.test = f.read()
            self.n_test = len(self.test)
        print(f"sizes of train, val, test = {self.n_train}, {self.n_val}, {self.n_test}")
        self.n_context = config['n_context'] # used for sample generation
        # encoding by character
        unique_chars = list(set(self.train))
        unique_chars.sort()
        vocab_size = len(unique_chars)
        print(f"vocab size = {vocab_size}, unique chars:\n{unique_chars}")
        self.encode_table = {c:i for i,c in enumerate(unique_chars)}
        self.decode_table = {i:c for i,c in enumerate(unique_chars)}
        self.x_train = self.encode(self.train).to(DEVICE)
        self.x_val = self.encode(self.val).to(DEVICE)
        self.x_test = self.encode(self.test).to(DEVICE)
    def encode(self, t):
        """
        t is a 1d string
        """
        return torch.tensor([self.encode_table[t1] for t1 in t], dtype=int)
    def decode(self, x):
        """
        x is a 1d integer tensor
        """
        return ''.join([self.decode_table[x1] for x1 in x])
    def draw(self,batch_size=1,partition='train'):
        """
        draw random sample with given batch size
        """
        if partition=='train':
            d, n = self.x_train, self.n_train
        elif partition=='val':
            d, n = self.x_val, self.n_val
        elif partition=='test':
            d, n = self.x_test, self.n_test
        else:
            raise ValueError("partition undefined!")
        # change from notebook: use randint
        # also remove device - somehow this increases performance.
        ids = torch.randint(n-self.n_context, (batch_size,))#, device=DEVICE)
        x = torch.stack([d[i:i+self.n_context] for i in ids])
        y = torch.stack([d[i+1:i+self.n_context+1] for i in ids])
        return x, y

model_data = ModelData()

@torch.no_grad()
def generate_from_model(model, n_generate=512, context=None, model_data=model_data):
    """
    context: a 1d string with size > n_context
    """
    if context is None:
        # draw a random context
        x = model_data.draw(1,'val')[0][0].tolist()
    else:
        x = model_data.encode(context).tolist()
    n_start = len(x)
    n_context = model_data.n_context
    for i in range(n_generate):
        x.append(int(model.generate(torch.tensor(x[-n_context:], device=DEVICE))))
    s = model_data.decode(x)
    NOT_USED = '\033[37m' # Light grey
    RESET = '\033[0m'  # Reset to default color
    print(NOT_USED + s[:n_start] + RESET + s[n_start:])
    return s

@torch.no_grad()
def validation(model, n_batches=1, batch_size=256, model_data=model_data, print_loss=False, **kwargs):
    """
    compute training and validation loss
    update from notebook version: new kwargs passed to model.forward()
    """
    # a total of n_batches*batch_size samples (and each sample gives n_context predictions)
    model.eval()
    losses_train = []
    losses_val = []
    for i in range(n_batches):
        x, y = model_data.draw(batch_size,'train')
        losses_train.append(model(x, y, **kwargs)[1].item())
        x, y = model_data.draw(batch_size,'val')
        losses_val.append(model(x, y, **kwargs)[1].item())
    model.train()
    if print_loss:
        print(f"epoch {model.current_epoch}, train loss = {np.mean(losses_train):.4f}, val loss = {np.mean(losses_val):.4f}")
    return np.mean(losses_train), np.mean(losses_val)