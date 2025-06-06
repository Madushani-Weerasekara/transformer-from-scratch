import torch
import torch.nn as nn
from torch.utils.data import Dataset, Dataloader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer # The class that will train the tokenizer(That will create the vocabulary giving the list of sentences)
from tokenizers.pre_tokenizers import Whitespace # Split the word by white space

from pathlib import Path # Allows us to create absolute path giving relative paths and 

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang] # Each item in the dataset is a pair of sentence(One in English and one in Italian) but we want to extract on particular language.

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang)) # The file where we'll be save the tokenizer
    if not Path.exists(tokenizer_path): # If tokenizer_path doesn't exists we create it
        tokenizer = Tokenizer(WordLevel(unk_token=['UNK'])) # If the tikenizer doesn't recognize in it's vocabulary replace it with this "UNK" Unknown
        tokenizer.pre_tokenizer = Whitespace()  # Split by white space
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2) # Build the trainer to train our tokenizer
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer) # Train the tokenizer 
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# Load the dataset
def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train') # Load the dataset and make configurable to change the language

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer (config, ds_raw, config['lang_src']) # For source language
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt']) # For taget language

    # Keep 90% training & 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = int(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
