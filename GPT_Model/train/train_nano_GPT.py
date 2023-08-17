#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:57:10 2023

@author: michiundslavki
"""

import torch
#from PyPDF2 import PdfReader
import glob
import torch.nn as nn
from torch.nn import functional as F
import numpy as np 

batch_size = 64
block_size = 256
max_iter = 5000
eval_intervall = 100
eval_iter = 200
vocab_size = 8 
n_embed = 384
n_head =  6
n_layer = 6
learn_rate = 3e-4

dropout = 0.2


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# read it in to inspect it
with open('/Users/michiundslavki/Documents/Data Science/NanoGPT/Data/Shakespeare_complete.txt', 'r', encoding='utf-8') as f:
    text = f.read()
new_text = text[900:]

#print(new_text[:1000])
# number of unique characters
unique_character = list(set(new_text))
#print(''.join(unique_character))

# create a mapping from characters to integers
stoi = { u_ch:i for i,u_ch in enumerate(unique_character) }
itos = { i:u_ch for i,u_ch in enumerate(unique_character) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# encode the entrire dataset and convert it to a torch tensor

data = torch.tensor(encode(new_text), dtype=torch.long)
#print(data.shape, data.dtype)

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
# define a function that gets a batch of blocksized tensors
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

#define a function to evaluate the model
@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.head_size = head_size
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * self.head_size**(-0.5)
        
        #wei = torch.zeros(T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
class Multi_Head(nn.Module):
    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out 
    
class Feed_Forward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, 4*n_embed),
                    nn.ReLU(),
                    nn.Linear(4*n_embed, n_embed),
                    nn.Dropout(dropout)
                )
    
    def forward(self, x):
        return self.net(x)

class Blocks(nn.Module):
    def __init__(self,n_embed, num_head):
        super().__init__()
        head_size = n_embed//num_head
        self.mha = Multi_Head(num_head, head_size)
        self.ffd = Feed_Forward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffd(self.ln2(x))
        return x
    
        
        
    
class GPT_Model(nn.Module):
    

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.pos_emedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.Block = nn.Sequential(*[Blocks(n_embed, num_head=n_head) for _ in range(n_layer)])
        # self.ffd = Feed_Forward((n_embed))
        # self.sa_heads = Multi_Head(4, n_embed//4)
        # self.Block = nn.Sequential(
        #     Blocks(n_embed, num_head=4),
        #     Blocks(n_embed, num_head=4),
        #     Blocks(n_embed, num_head=4),
        #     nn.LayerNorm(n_embed)
        #     )
        
    
    def forward(self, idx, targets=None):
        B, T =idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embed = self.token_embedding_table(idx) # (B,T,C)
        pos_embed = self.pos_emedding_table(torch.arange(T, device=device))
        x = token_embed + pos_embed
        #x = self.sa_heads(x)
        #x = self.ffd(x)
        x = self.Block(x)
        logits = self.lm_head(x) #(B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

#train the model 
vocab_size = len(unique_character)
model = GPT_Model()
m = model.to(device)
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learn_rate)


for iter in range(max_iter): # increase number of steps for good results... 
    

    if iter % eval_intervall == 0:
        losses = estimate_loss()
        print(f"after {iter} iterations, train loss: {losses['train']} and the eval loss: {losses['val']}")
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
