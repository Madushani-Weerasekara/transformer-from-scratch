import torch
import torch.nn as nn
from torch.utils.data import Dataset, Dataloader, random_split

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from  config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer # The class that will train the tokenizer(That will create the vocabulary giving the list of sentences)
from tokenizers.pre_tokenizers import Whitespace # Split the word by white space

from torch.utils.tensorboard import SummaryWriter
import warnings

from tqdm import tqdm

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

# Building the model
def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

# build the training loop
def train_model(config):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True) # Make sure that weights folder is created

    # Load the dataset
    train_dataloader, val_dataloader, Tokenizer_src, Tokenizer_tgt = get_ds(config)
    model = get_model(config, Tokenizer_src.get_vocab_size(), Tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename) # load the file
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=Tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # finally the training  loop
    for epoch in  range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            # Tensors
            encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch, 1, seq_len, seq_len)

            # Run the tensors through transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # Calculate output of the encoder (bath, seq_len,  d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # Calculate output of the decoder (bath, seq_len,  d_model)
            proj_output = model.projet(decoder_output) # Projection output (batch, seq_len, d_model)

            label = batch['label'].to(device) # Extract the label from the batch (batch, seq_len)

            # (batch, seq_len, d_model) --> (batch *  seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, Tokenizer_tgt.get_vocab_size())), label.view(-1)

            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"}) # Update progress bar

            # Log the loss
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            # Save the model at end of the every epoch
            model_filename = get_weights_file_path(config, f'{epoch: 02d}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(), # All the weights of the model
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)

if __name__ == '__main__':
    Warning.filterwarnings('ignore')
    config = get_config()
    train_model(config)



