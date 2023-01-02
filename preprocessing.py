import json
import pandas as pd
from tokenizers import Tokenizer


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
        tokenizer = Tokenizer(AWD_LSTM)
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


