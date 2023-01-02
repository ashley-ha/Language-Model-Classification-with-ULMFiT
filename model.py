import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from transformers import BertModel 
from torchtext.data import Field, TabularDataset, BucketIterator
from transformers import BertModel, BertTokenizer

# Define the fields for the dataset
text_field = Field(tokenize='spacy', include_lengths=True)
label_field = Field(sequential=False, use_vocab=False)

# Load the dataset
train_dataset, valid_dataset, test_dataset = TabularDataset.splits(
    path='path/to/dataset',
    train='train.tsv',
    validation='valid.tsv',
    test='test.tsv',
    fields=[('text', text_field), ('label', label_field)],
    skip_header=True
)

# Build the vocabulary
text_field.build_vocab(train_dataset, max_size=50000)

# Define the dataloaders
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, valid_dataset, test_dataset),
    batch_size=32,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device='cuda' if torch.cuda.is_available() else 'cpu')

class ULMFiT(nn.Module):
    def __init__(self, pre_trained_model, num_classes):
        super().__init__()
        self.pre_trained_model = pre_trained_model
        self.num_classes = num_classes
        self.fc = nn.Linear(pre_trained_model.hidden_size, num_classes)
    
    def forward(self, input_text):
        # Pass input text through the pre-trained model to get hidden states
        hidden_states = self.pre_trained_model(input_text)
        
        # Use the last hidden state as the input to the classifier
        logits = self.fc(hidden_states[-1])
        return logits

def fine_tune_model(model, train_iterator, valid_iterator, criterion, optimizer, scheduler, num_epochs):
    model.train()
    
    for epoch in range(num_epochs):
        train_loss = 0
        valid_loss = 0
        
        # Set the learning rate for this epoch using the scheduler
        scheduler.step()
        
        for batch in train_iterator:
            optimizer.zero_grad()
            logits = model(batch.text)
            loss = criterion(logits, batch.label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        for batch in valid_iterator:
            logits = model(batch.text)
            loss = criterion(logits, batch.label)
            valid_loss += loss.item()
        
        train_loss /= len(train_iterator)
        valid_loss /= len(valid_iterator)
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.3f}, Valid Loss: {valid_loss:.3f}')

def main():

    # Load your pre-trained model
    pre_trained_model = BertModel.from_pretrained('bert-large-uncased-whole-word-masking')
    
    # Define the number of classes for your classification task
    num_classes = 2
    
    # Create the ULMFiT model
    model = ULMFiT(pre_trained_model, num_classes)
    
    # Define your training and validation datasets and dataloaders
    #train_iterator, valid_iterator, test_iterator = 
    
    # Define your loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Define a learning rate scheduler
    scheduler = optim.lr_scheduler.SlantedTriangularLR(optimizer)
    
    # Fine-tune the model
    fine_tune_model(model, train_iterator, valid_iterator, criterion, optimizer, scheduler, num_epochs=10)
    
    # Evaluate the model on the test set
    model.eval()
    test_loss = 0
    correct = 0
    for batch in test_iterator:
        logits = model(batch.text)
        loss = criterion(logits, batch.label)
        test_loss += loss.item()
        correct += (torch.argmax(logits, dim=1) == batch.label).sum().item()
    test_loss /= len(test_iterator)
    accuracy = correct / len(test_iterator.dataset)
    print(f'Test Loss: {test_loss:.3f}, Accuracy: {accuracy:.3f}')

