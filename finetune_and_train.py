 # Note that the fine-tuning & training steps are using only 1 epoch each for demonstration purposes here, 
 # The number of epochs should be adjusted based on the requirements & available computational resources.

from fastai.text import *
import os

def fine_tune_language_model(data_dir, lm_dir, bs=48, epochs=1):
    # Load text data for fine-tuning
    data_lm = TextLMDataBunch.from_folder(data_dir, bs=bs)

    # Initialize the AWD-LSTM language model with the pre-trained encoder (fastai)
    learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
    learn.load_encoder(os.path.join(lm_dir, 'pretrained_lm_encoder'))

    # Fine-tune lm
    learn.freeze()
    learn.fit_one_cycle(epochs, 1e-2)

    # Save the fine-tuned language model
    learn.save_encoder(os.path.join(lm_dir, 'fine_tuned_lm_encoder'))


def train_classifier(data_dir, lm_dir, bs=48, epochs=1):
    # Load text data for training the classifier
    data_clas = TextClasDataBunch.from_folder(data_dir, bs=bs)

    # Initialize the AWD-LSTM classifier with the fine-tuned encoder
    learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
    learn.load_encoder(os.path.join(lm_dir, 'fine_tuned_lm_encoder'))

    # Train the classifier
    learn.fit_one_cycle(epochs, 1e-2)

    # Save the trained classifier
    learn.save(os.path.join(lm_dir, 'trained_classifier'))


if __name__ == '__main__':
    data_path = './data/imdb'  # Path to your dataset
    lm_path = './lm'  # Path to the saved language model

    # Fine-tune lm
    fine_tune_language_model(data_path, lm_path)

    # Train classifier
    train_classifier(data_path, lm_path)
