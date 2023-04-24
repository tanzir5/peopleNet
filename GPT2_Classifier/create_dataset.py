from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import GPT2Config
import numpy as np
import evaluate
import pandas as pd

def get_text_label(data_idx, data, target_text):
  for col_idx, content in enumerate(data['table']['content']):
      splitted_words = content.split()
      if 'aged' in splitted_words:
        aged_idx = splitted_words.index('aged')
        if (aged_idx != len(splitted_words)-1 and 
            splitted_words[aged_idx+1].isnumeric() and
            data['table']['column_header'][col_idx] == 'death_date'):
          return target_text[data_idx], int(splitted_words[aged_idx+1])
  return None, None

def mask_number(text):
  masked_text = ""
  for i, char in enumerate(text):
    if char.isdigit():
      if len(masked_text) == 0 or masked_text[-1] != '*':
        new_char = '*'
      else:
        new_char = ''
    else:
      new_char = char
    masked_text += new_char
  return masked_text

def create_csv(dataset, path):
  texts = []
  masked_texts = []
  labels = []
  input_text = dataset['input_text'] 
  target_text = dataset['target_text'] 
  for data_idx, data in enumerate(input_text):
    text, label = get_text_label(data_idx, data, target_text)
    if text is not None:
      texts.append(text)
      masked_texts.append(mask_number(text))
      labels.append(float(label))  
  df = pd.DataFrame({'text':texts, 'masked_text':masked_texts, 'label':labels})
  df.to_csv(path)



dataset = load_dataset("wiki_bio")
create_csv(dataset['train'], 'wiki_train.csv')
create_csv(dataset['val'], 'wiki_val.csv')
create_csv(dataset['test'], 'wiki_test.csv')


