from model import AWD_LSTM
from preprocessing import Preprocessing
import json

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
#Tokenize the text
questions_tokens, answers_tokens = Preprocessing.tokenize_text(questions, answers)    

#Create the vocabulary       
vocab_size, word_index = Preprocessing.create_vocab(questions_tokens, answers_tokens)

#Convert to numerical
questions_numerical, answers_numerical = Preprocessing.convert_to_numerical(questions_tokens, answers_tokens)

