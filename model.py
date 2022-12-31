import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

# import the necessary libraries for ULMFiT from fastai
from fastai.text.all import * 
from fastai.text.learner import *


class AWD_LSTM(nn.Module):
    def __init__(self, vocab_sz: int, emb_sz: int, n_hid: int, n_layers: int, pad_token: int,
    hidden_p: float, input_p: float, embed_p:float, weight_p:float, qrnn:bool=False, bidir:bool=False):
        super().__init__()
        self.vocab_sz = vocab_sz
        self.emb_sz = emb_sz
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.pad_token = pad_token
        self.hidden_p = hidden_p
        self.input_p = input_p
        self.embed_p = embed_p
        self.weight_p = weight_p
        self.qrnn = qrnn
        self.bidir = bidir

        # Define the embedding layer
        self.embedding = nn.Embedding(self.vocab_sz, self.emb_sz, self.pad_token)

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.emb_sz, self.n_hid, self.n_layers, batch_first=True, bidirectional=self.bidir)

        # Define the dropout layers
        self.input_dp = nn.Dropout(self.input_p)
        self.hidden_dp = nn.Dropout(self.hidden_p)
        self.embed_dp = nn.Dropout(self.embed_p)
        
        # Define the fully-connected layer
        self.fc = nn.Linear(self.n_hid, self.vocab_sz)

    def forward(self, x):
        # Apply the embedding layer
        x = self.embedding(x)
        x = self.embed_dp(x)
        
        # Apply the LSTM layer
        x, (hidden, cell) = self.lstm(x)


