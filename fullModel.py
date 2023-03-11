# Import Libraries
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

def main():
    BATCH_SIZE = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 5
    N_LAYERS = 3
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    EMBED_DIM = 32
    HIDDEN_DIM = 64
    OUTPUT_DIM = 1
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    # Load in data
    TEXT = torchtext.legacy.data.Field(tokenize='spacy', batch_first=True)
    LABEL = torchtext.legacy.data.LabelField(dtype=torch.float)
    train_data, test_data = text_classification.DATASETS['AG_NEWS'](root='./data', ngrams=2, vocab=Vocab(min_freq=3))
    train_data, val_data = train_data.split(split_ratio=0.7)

    TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(train_data)

    train_iter, val_iter, test_iter = torchtext.legacy.data.BucketIterator.splits((train_data, val_data, test_data), batch_size=BATCH_SIZE, device=device)

    # Initialize the model
    model = TextSentiment(len(TEXT.vocab), EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    # Define the loss and optimizer functions
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    model = model.to(device)
    criterion = criterion.to(device)

    # Train and evaluate model
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iter, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_iter, criterion, device)
        end_time = time.time()

        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'ag_news_model.pt')

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%')

    # Testing the model
    model.load_state_dict(torch.load('ag_news_model.pt'))
    test_loss, test_acc = evaluate(model, test_iter, criterion, device)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

