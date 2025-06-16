import torch
import torch.nn as nn
from torch.utils.data import Dataset, Dataloader, random_split

from dataset import BilingualDataset, causal_mask

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

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids)) # Maximum length of source language
        max_len_tgt = max(max_len_tgt, len(tgt_ids)) # Maximum length of the target langugae

    print(f'Maximum length of the source sentnece : {max_len_src}')
    print(f'Maximum length of the target sentence : {max_len_tgt}')

    # Create the data loaders
    train_dataloader = Dataloader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = Dataloader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt