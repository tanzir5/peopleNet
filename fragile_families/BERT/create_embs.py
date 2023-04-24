from transformers import RobertaModel
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from tokenizers.implementations import ByteLevelBPETokenizer
from datasets import load_dataset
from transformers import pipeline
from transformers import RobertaTokenizerFast
import numpy as np
import torch
from tokenizers.processors import BertProcessing
from tokenizers.implementations import ByteLevelBPETokenizer
from tqdm import tqdm
import pandas as pd

def tokenize_function(text):
    
    return tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_seq_length,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
    )

def get_emb(texts):
  input_ids = torch.tensor(tokenize_function(texts)['input_ids']).to(device)
  #print(input_ids.size())
  out = model(input_ids=input_ids, output_hidden_states=True)
  #print(len(out))
  #print(out)
  hidden_states = out[0].detach()
  #print(type(hidden_states))
  #print(hidden_states.shape)
  #print(out)
  emb = torch.mean(hidden_states, dim=1)
  #print(emb.shape)
  return emb.detach().cpu().numpy()

max_seq_length=512

tokenizer = RobertaTokenizerFast(tokenizer_file="byte-level-BPE.tokenizer.json", max_len=512)
special_tokens_dict = {'additional_special_tokens': ["<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
    "<sep>"]}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

model = RobertaModel.from_pretrained('./final_models/')
print(sum(p.numel() for p in model.parameters()))
exit(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()
data_f = open("background_split_by_lines_2.txt")
lines = data_f.readlines()
embs = []
family_ids = []
batch_lines = []
for i,line in tqdm(enumerate(lines)):
  current_family = int((i/2000))
  family_ids.append(current_family)
  batch_lines.append(line)
  if((i == len(lines)-1 or i%100 == 0) and len(batch_lines) > 0):
    batch_embs = get_emb(batch_lines)
    batch_lines = []
    for emb in batch_embs:
        embs.append(emb)
  
  
df = pd.DataFrame({'family_id': family_ids, 'emb': embs})
df.to_csv('background_embs_x2000.csv', index=False)