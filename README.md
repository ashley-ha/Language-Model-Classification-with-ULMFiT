# Language Model Implementation with ULMFiT

This repository contains a PyTorch implementation of Universal Language Model Fine-tuning (ULMFiT), a powerful and efficient transfer learning method that can be applied to various tasks in NLP. The method is described in detail in the research paper Universal Language Model Fine-tuning for Text Classification. https://arxiv.org/abs/1801.06146

About ULMFiT
For a more comprehensive look at ULMFiT, check out my Medium post: https://medium.com/@ashleyha/lets-learn-about-universal-language-model-fine-tuning-ulmfit-through-practical-applications-fea0aed2cf96.


Overview
Universal Language Model Fine-tuning (ULMFiT) is a technique for fine-tuning a pre-trained language model for a specific natural language processing (NLP) task. It was introduced by Jeremy Howard and Sebastian Ruder in a paper published in 2018.

ULMFiT involves fine-tuning a * pre-trained * language model on a labeled dataset for a specific NLP task, such as sentiment analysis or named entity recognition (NER). The fine-tuning process involves adjusting the model's parameters to better fit the characteristics of the new task. This is done by first fine-tuning the language model on a large, general-purpose dataset & then fine-tuning it on the task-specific dataset (AG News dataset in this case).

ULMFiT has been shown to be effective at improving the performance of NLP models on a wide range of tasks, and has become a popular approach in the field of NLP. It has been used to achieve state-of-the-art results on several benchmarks and has been widely adopted in industry and academia.

## Adaptations for my specific use cases from the original research paper:
### Data
- I used the latest AG News dataset for the text classification task

## My PyTorch implementation follows the ULMFiT approach as described in the paper and consists of three main steps:

1. Pretrain the language model: I used the AWD-LSTM model architecture as a language model and pretrained it on various datasets such as Wikipedia, Yelp, etc.

2. Fine-tune the language model: I fine-tuned the pre-trained language model on the AG News dataset. This involves using discriminative learning rates, where a smaller learning rate is used for the earlier layers of the model and a larger learning rate for the later layers.

3. Train the classifier: I used the fine-tuned language model as a feature extractor for the text classification task by passing the input text through the model and using the resulting hidden states as features for a classification model. I employed the TextSentiment class as the classifier.

## Usage
Model Architectures 
```
import torch 
import torch.nn as nn
import torch.optim as optim 
from fastai.text import *
```
Loading in your data via path
```
data_lm = TextLMDataBunch.from_csv(your_path, csv_name='your_csvname', valid_pct=0.2)
```
In the main() function, I first load the AG News dataset and split it into training, validation, and test sets using the text_classification module of TorchText. I then built the vocabulary for the TEXT field and the label for the LABEL field. I used the BucketIterator class to create data iterators for the three sets, which I used in training, validation, and testing.

Next, I initialize the ULMFiT model by creating an instance of the TextSentiment class and setting its attributes based on the hyperparameters defined at the beginning of the function. I also load the pre-trained GloVe embeddings for the vocabulary and set them as the initial weights for the embedding layer of the model.
I define the loss function and optimizer, move the model and loss function to the device (either GPU or CPU), and then train and evaluate the model for the specified number of epochs. During training, I save the model state whenever the validation loss improves.


``` Sample Model output:
Epoch: 01 | Epoch Time: 0m 26s
	Train Loss: 0.317 | Train Acc: 86.41%
	 Val. Loss: 0.162 |  Val. Acc: 94.31%
Epoch: 02 | Epoch Time: 0m 26s
	Train Loss: 0.144 | Train Acc: 95.11%
	 Val. Loss: 0.127 |  Val. Acc: 95.65%
Epoch: 03 | Epoch Time: 0m 27s
	Train Loss: 0.096 | Train Acc: 96.87%
	 Val. Loss: 0.104 |  Val. Acc: 96.35%
Epoch: 04 | Epoch Time: 0m 27s
	Train Loss: 0.070 | Train Acc: 97.69%
	 Val. Loss: 0.090 |  Val. Acc: 96.82%
Epoch: 05 | Epoch Time: 0m 27s
	Train Loss: 0.054 | Train Acc: 98.23%
	 Val. Loss: 0.084 |  Val. Acc: 97.09%
Test Loss: 0.080 | Test Acc: 97.23%
Confusion Matrix:
[[2265  105   40   90]
 [  55 2379   27   39]
 [  51   32 2372   45]
 [  68   52   38 2342]] 

``` 

This implementation demonstrates how to apply ULMFiT to a text classification task using PyTorch. Other researchers can use this code as a starting point for their own projects, adapting it as necessary to accommodate different datasets or NLP tasks.

# Contact 
Feel free to reach out to me regarding issues, data, feedback (is welcome!) & requests relating to my code implementation:

Ashley Ha - ashleyha@berkeley.edu | https://medium.com/@ashleyha | https://www.linkedin.com/in/ashleyeastman/ | https://ashleyha.substack.com/
