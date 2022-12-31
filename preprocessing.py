import json
import pandas as pd
from keras.preprocessing.text import Tokenizer


# Preprocessing steps before testing the model with AWD_LSTM class 
# 1) Tokenizing the text
# 2) Creating a vocabulary
# 3) converting the text to numerical format
class Preprocessing:
    def __init__(self, questions, answers, questions_tokens, answers_tokens):
        self.questions = questions
        self.answers = answers
        self.questions_tokens = questions_tokens
        self.answers_tokens = answers_tokens

    def tokenize_text(questions, answers):
        """
        Tokenizes the questions and answers.
        
        Parameters
        ----------
        questions : list
            A list of questions.
        answers : list
            A list of answers.
        
        Returns
        -------
        tuple
            A tuple containing the tokenized questions and answers.
        """
        # Tokenize the questions and answers
        tokenizer = Tokenizer()
        questions_tokens = tokenizer.fit_on_texts(questions)
        answers_tokens = tokenizer.fit_on_texts(answers)
        
        return questions_tokens, answers_tokens

    def create_vocab(questions_tokens, answers_tokens):
        """
        Creates a vocabulary from the tokenized questions and answers.
        
        Parameters
        ----------
        questions_tokens : list
            A list of tokenized questions.
        answers_tokens : list
            A list of tokenized answers.
        
        Returns
        -------
        tuple
            A tuple containing the vocabulary size and the word index.
        """
        # Get the vocabulary size
        vocab_size = len(Tokenizer.word_index) + 1  # Add 1 to include the padding token
        
        # Get the word index
        word_index = Tokenizer.word_index
        
        return vocab_size, word_index

    def convert_to_numerical(questions_tokens, answers_tokens):
        """
        Converts the tokenized questions and answers into numerical format.
        
        Parameters
        ----------
        questions_tokens : list
            A list of tokenized questions.
        answers_tokens : list
            A list of tokenized answers.
        
        Returns
        -------
        tuple
            A tuple containing the numerical questions and answers.
        """
        # Convert the tokenized questions and answers into numerical format
        questions_numerical = Tokenizer.texts_to_sequences(questions_tokens)
        answers_numerical = Tokenizer.texts_to_sequences(answers_tokens)
        
        return questions_numerical, answers_numerical

# define the path to the dataset
full_path = 'YOURFULLPATHHERE'
# reading the JSON data using json.load()
file = 'train-v2.0.json'
# Load the dataset from a json file
with open(file, 'r') as f:
    dataset = json.load(f)

questions = []
answers = []
# Access the data in the dataset
for example in dataset['data']:
    # Each example consists of a context (the article text) and a list of questions and answers
    context = example['paragraphs']
    for paragraph in example['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            question = qa['question']
            if qa['answers']:  # Check if the answers list is not empty
                answer = qa['answers'][0]['text']  # There may be multiple answers, but we'll just use the first one
            else:
                answer = None
            questions.append(question)
            answers.append(answer)