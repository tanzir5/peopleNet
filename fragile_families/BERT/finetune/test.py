from transformers import RobertaTokenizerFast
from transformers import AutoModelForSequenceClassification

import copy
import evaluate
import numpy as np
import os
import pandas as pd
import torch

#TEST = "SMALL"
TEST = "MEDIUM"
#TEST = "ALL"
if TEST == "SMALL":
  background_path = '../small_background_x2000.csv'
else:
  background_path = '../background_x2000.csv'
test_path = '../leaderboard.csv'
train_path = '../train.csv'

CHALLENGE_ID = 'challengeID'
tasks_type = {
  "gpa":"regression",
  "grit":"regression",
  "materialHardship":"regression",
  "eviction":"binary",
  "layoff":"binary",
  "jobTraining":"binary"
}

BATCH_SIZE = 48
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

label_name='grit'
train_labels = np.array(pd.read_csv(train_path)[label_name].tolist())
train_labels = train_labels[~np.isnan(train_labels)]
train_mean = np.mean(train_labels)
print(train_mean)
#exit(0)
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

def load_model(label):
  all_model_paths = [x for x in os.listdir('./'+label+'/') if x.startswith('check')]
  all_model_paths.sort(reverse=True, key=lambda x: (len(x), x))
  print(all_model_paths)
  model = AutoModelForSequenceClassification.from_pretrained("./"+label+"/"+all_model_paths[2])
  return model

def run_data(model, challenge_id, background_encoded_df):
  #print("Hello", challenge_id)
  #print(background_encoded_df[CHALLENGE_ID])
  df = copy.deepcopy(background_encoded_df[background_encoded_df[CHALLENGE_ID] == challenge_id])
  df = df.sample(n=10)
  input_ids = torch.tensor(df['input_ids'].tolist()).to(DEVICE)
  outputs = model(input_ids)
  print(outputs)
  outputs = outputs['logits'].detach().numpy()
  print("output shape", outputs.shape)
  return np.mean(outputs)

def eval_results(predictions, golds, task, label_name):
  if task == 'binary':
    pass
  else:
    pass
  train_labels = np.array(pd.read_csv(train_path)[label_name].tolist())
  train_labels = train_labels[~np.isnan(train_labels)]
  train_mean = np.mean(train_labels)
  R = 1 - np.sum(np.square(predictions - golds)) / np.sum(np.square(prediction - train_mean))
  return R

def eval_task(main_test_df, background_encoded_df, target_label):
  model = load_model(target_label)
  predictions = []
  golds = []
  for i in range(len(main_test_df)):
    if pd.notna(main_test_df.iloc[i][target_label]):
      current_idx = main_test_df.iloc[i][CHALLENGE_ID]
      print("gold", main_test_df.iloc[i][target_label])
      current_prediction = run_data(model, current_idx, background_encoded_df)
      predictions.append(current_prediction)
      golds.append(main_test_df.iloc[i][target_label])
  results = eval_results(predictions, golds, tasks_type[target_label])
  print(results)
  f.write(target_label + '\n')
  f.write(str(results))
  f.write("\n")

max_seq_length = 512
tokenizer = RobertaTokenizerFast(tokenizer_file="../byte-level-BPE.tokenizer.json", max_len=max_seq_length)

main_test_df = pd.read_csv(test_path)
background_df = load_background(background_path)
background_encoded_df = tokenize_background(background_df, tokenizer)

for column_name in main_test_df.columns:
  if column_name in tasks_type:
    print("Working for", column_name)
    eval_task(main_test_df, background_encoded_df, column_name)
