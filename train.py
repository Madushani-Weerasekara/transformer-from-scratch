import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer # The class that will train the tokenizer(That will create the vocabulary giving the list of sentences)
from tokenizers.pre_tokenizers import Whitespace # Split the word by white space
