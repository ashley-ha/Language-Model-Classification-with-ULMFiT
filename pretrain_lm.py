from fastai.text import *
import os

def pretrain_language_model(data_dir, lm_dir, bs=48, epochs=1):
    # Load in our text data:
    data_lm = TextLMDataBunch.from_folder(data_dir, bs=bs)

    # Initialize the AWD-LSTM language model
    learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)

    # Pretraining our language model
    learn.fit_one_cycle(epochs, 1e-2)

    # Save the pre-trained language model
    if not os.path.exists(lm_dir):
        os.makedirs(lm_dir)
    learn.save_encoder(os.path.join(lm_dir, 'pretrained_lm_encoder'))

if __name__ == '__main__':
    data_path = './data/imdb'  # Path to your dataset
    lm_path = './lm'  # Path to save the pre-trained language model
    pretrain_language_model(data_path, lm_path)
