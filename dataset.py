import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype=torch.int64) # Start Of Sentence tokens
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype=torch.int64) # End Of Sentence tokens
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype=torch.int64) # Padding tokens

        def __len__(self):
            return len(ds) # Return the length of the dataset from Hugging face
        
        def __getitem(self, index: any) -> any:
            src_tgt_pair = self.ds[index] # Extract the original pair from the hugging face
            src_text = src_tgt_pair['translation'][self.src_lang] # Extract the source text
            tgt_text = src_tgt_pair['translation'] [self.tgt_lang] # Extract the target text

            # This is done by the encode method & the decode method, this gives us the input IDs
            enc_input_tokens = self.tokenizer_src.encode(src_text).ids # convert encoder input tokens to IDs
            dec_input_tokens = self.tokenizer_tgt.decode(tgt_text).ids # convert decoder input tokens to IDs

            enc_num_padding_tokens = seq_len - len(enc_input_tokens) - 2 # Calculate the number of padding tokens for the encoder side
            dec_num_padding_tokens = seq_len - len(dec_input_tokens) -1 # Calculate the number of padding tokens for the decoder side

            if enc_input_tokens < 0 or dec_input_tokens < 0:
                raise ValueError('Sentence is too long')
            
            # Build tensors for encoder input  
            # Add SOS & EOS to the source text
            encoder_input = torch.cat(
                [
                    self.sos_token, # SOS token
                    torch.tensor(enc_input_tokens, dtype=torch.int64), # Token of the source text
                    self.eos_token, # EOS token
                    torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64) # Add enough padding tokens to sequence length
                ]
            )

            # Build the tensors for the decoder input
            decoder_input = torch.cat(
                [
                    self.sos_token, # SOS token
                    torch.tensor(dec_input_tokens, dtype=torch.int64),  
                    torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64) # Add enough padding tokens to the sequence length

                ]
            )

            # Build the tensor for the label/target
            