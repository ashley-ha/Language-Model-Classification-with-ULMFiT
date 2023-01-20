# Ultra-Classification-with-ULMFiT

Pytorch code implementation of the Universal Language Model Fine-tuning (ULMFiT), a powerful and efficient transfer learning method that can be applied to any task in NLP described in detail in the research paper [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)

## About ULMFiT
For a more comprehensive look at ULMFiT, check out my [Medium post](https://medium.com/p/fea0aed2cf96)

### Overview
Universal Language Model Fine-tuning (ULMFiT) is a technique for fine-tuning a pre-trained language model for a specific natural language processing (NLP) task. It was introduced by Jeremy Howard and Sebastian Ruder in a paper published in 2018.

ULMFiT involves fine-tuning a pre-trained language model on a labeled dataset for a specific NLP task, such as sentiment analysis or named entity recognition. The fine-tuning process involves adjusting the model's parameters to better fit the characteristics of the new task. This is done by first fine-tuning the language model on a large, general-purpose dataset and then fine-tuning it on the specific task-specific dataset.

ULMFiT has been shown to be effective at improving the performance of NLP models on a wide range of tasks, and has become a popular approach in the field of NLP. It has been used to achieve state-of-the-art results on a number of benchmarks, and has been widely adopted in industry and academia.

## Differences for my specific use cases from the original research paper:
### Data
- I used the latest DBpedia wiki datasets which includes the Long abstracts from Wikipedia articles
- SFU Review Corpus - a collection of movie, book, and consumer product reviews 
- [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/) - a reading comprehension dataset consisting of questiosn posed by crowdworkers on a set of Wiki articles, where the answer to every question is a segement of text, or span, from the corresponding reading passage 

## My PyTorch Implementation 
1. First, I needed to obtain a pre-trained language model. You can either use a pre-trained model provided by Hugging Face, or you can train your own language model on a large dataset such as Wikipedia, Yelp, etc. I chose a number of various datasets that you can see above.

2. Next, I needed to fine-tune the pre-trained language model on my specific dataset. This involves using a smaller learning rate for the earlier layers of the model, and a larger learning rate for the later layers. This is known as discriminative learning rates.

3. To further improve my model's performance, I used slanted triangular learning rates, which Jeremy and Sebastian go into detail on in their research paper. This involves gradually increasing the learning rate over the course of training, with a sharp decrease at the end.

4. Finally, I used the fine-tuned language model as a feature extractor for your text classification task. Simply pass the input text through the model and use the resulting hidden states as features for a classification model.

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
This Repo is currently in progress ! Last updated 1/10/23
# Contact 
Feel free to reach out to me regarding issues, data, feedback (is welcome!) & requests relating to my code implementation:

Ashley Ha - ashleyha@berkeley.edu | https://medium.com/@ashleyha | https://www.linkedin.com/in/ashleyeastman/
