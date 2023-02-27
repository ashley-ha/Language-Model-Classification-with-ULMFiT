import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import text_classification
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import time

class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        embedded = self.dropout(embedded)
        packed_output, (hidden, cell) = self.encoder(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1) if self.encoder.bidirectional else hidden[-1,:,:])
        output = self.fc(hidden)
        return output


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in tqdm(iterator):
        text, offsets, label = batch
        text, offsets, label = text.to(device), offsets.to(device), label.to(device)
        optimizer.zero_grad()
        predictions = model(text, offsets)
        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in tqdm(iterator):
            text, offsets, label = batch
            text, offsets, label = text.to(device), offsets.to(device), label.to(device)
            predictions = model(text, offsets)
            loss = criterion(predictions, label)
            acc = binary_accuracy(predictions, label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc
