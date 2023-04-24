import copy
import evaluate
import numpy as np
import os
import pandas as pd

from datasets import Dataset
from datasets import load_dataset
#from transformers import RobertaTokenizerFast
from transformers import BertTokenizerFast
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification

BINARY_PRETRAINED = True
#BINARY = False


#TEST = "SMALL"
TEST = "MEDIUM"
#TEST = "ALL"
if TEST == "SMALL":
  train_path = '../small_train.csv'
  background_path = '../small_background_x2000.csv'
else:
  train_path = '../train.csv'
  if BINARY_PRETRAINED:
    background_path = '../binary/background_x2000.csv'
  else:
    background_path = '../background_x2000.csv'

if BINARY_PRETRAINED is False:
  tokenizer = RobertaTokenizerFast(tokenizer_file="../byte-level-BPE.tokenizer.json", max_len=max_seq_length)
  model_path = '../final_models'
else:
  tokenizer = BertTokenizerFast.from_pretrained('../binary/tokenizer')
  model_path = '../binary/final_models'

def get_stats(a):
  return np.min(a), np.median(a), np.max(a), np.mean(a), np.std(a)

def compute_regression_metrics(eval_pred):
  metric = evaluate.load("mae")
  predictions, labels = eval_pred
  print(get_stats(predictions))
  return metric.compute(predictions=predictions, references=labels)

def compute_classification_metrics(eval_pred):
  metric = evaluate.load("accuracy")
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)
  print(get_stats(predictions))
  return metric.compute(predictions=predictions, references=labels)


def get_train_dict(main_train_df, background_encoded_df, label_name):
  idxs = []
  input_ids = []
  labels = []
  for i in range(len(main_train_df)):
    if pd.notna(main_train_df.iloc[i][label_name]):
      current_idx = main_train_df.iloc[i][CHALLENGE_ID]
      current_idx_rows = background_encoded_df[background_encoded_df[CHALLENGE_ID] == current_idx]
      current_input_ids = current_idx_rows['input_ids'].tolist()
      label = main_train_df.iloc[i][label_name]
      if tasks_type[label_name] == 'binary':
        label = int(label)
      current_labels = [label] * len(current_idx_rows)
      labels.extend(current_labels)
      input_ids.extend(current_input_ids)
      idxs.extend([current_idx]* len(current_idx_rows))
  return {
    CHALLENGE_ID: idxs, 
    'input_ids':input_ids, 
    'label':labels,
    }

def get_train_eval_dataset(main_train_df, background_encoded_df, label_name):
  train_dict = get_train_dict(main_train_df, background_encoded_df, label_name)
  train_ds = Dataset.from_dict(train_dict)
  train_ds = train_ds.shuffle(seed=42)
  ds = train_ds.train_test_split(test_size=0.1)
  train_ds = ds['train']
  eval_ds = ds['test']
  return train_ds, eval_ds

def train_task(main_train_df, background_encoded_df, label_name):
  train_ds, eval_ds = get_train_eval_dataset(main_train_df, background_encoded_df, label_name)
  if tasks_type[label_name] == 'regression':
    num_labels = 1
    compute_metrics = compute_regression_metrics
  else:
    num_labels = 2
    compute_metrics = compute_classification_metrics

  model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
  if BINARY_PRETRAINED:
    output_dir = '../binary/' + label_name
  else:
    output_dir = label_name
  if os.path.exists(output_dir) == False:
    os.mkdir(output_dir)
  training_args = TrainingArguments(
                    output_dir=output_dir, 
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    num_train_epochs=EPOCHS,
                    per_device_train_batch_size=BATCH_SIZE,
                    per_device_eval_batch_size=BATCH_SIZE,
                    save_total_limit=1,
                    load_best_model_at_end=True,
                    learning_rate=5e-5,
                    )
  trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_ds,
       eval_dataset=eval_ds,
       compute_metrics=compute_metrics,
   )
  print(train_ds)
  print(eval_ds)
  trainer.train()
  final_dir = output_dir+"/"+"final_models"
  if os.path.exists(final_dir) == False:
    os.mkdir(final_dir)
  trainer.save_model(final_dir)


def load_background(path):
  if TEST == "MEDIUM":
    df = pd.read_csv(path, skiprows=lambda x: x%100!=0)
    print("medium loaded, columns=", df.columns)
  else:
    df = pd.read_csv(path)
  df.rename(columns={'family_id': CHALLENGE_ID}, inplace=True)
  df[CHALLENGE_ID] = df[CHALLENGE_ID] + 1
  return df

def tokenize(texts, tokenizer):
  return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_seq_length,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
    )

def tokenize_background(df, tokenizer):
  encoded_text = tokenize(df['text'].tolist(), tokenizer)
  df['input_ids'] = encoded_text['input_ids']
  df['attention_mask'] = encoded_text['attention_mask']
  return df


CHALLENGE_ID = 'challengeID'
tasks_type = {
  "gpa":"regression",
  "grit":"regression",
  "materialHardship":"regression",
  "eviction":"binary",
  "layoff":"binary",
  "jobTraining":"binary"
}

EPOCHS = 10
BATCH_SIZE = 48
max_seq_length = 512

main_train_df = pd.read_csv(train_path)
background_df = load_background(background_path)
background_encoded_df = tokenize_background(background_df, tokenizer)

for column_name in main_train_df.columns:
  if column_name in tasks_type:
    print("Working for", column_name)
    train_task(main_train_df, background_encoded_df, column_name)
