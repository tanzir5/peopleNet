from transformers import RobertaTokenizerFast
from transformers import BertTokenizerFast
from transformers import AutoModelForSequenceClassification

import copy
import evaluate
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm

BINARY_PRETRAINED = True
#BINARY = False

test_path = '../../FFChallenge_v2/test.csv'
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
else:
  tokenizer = BertTokenizerFast.from_pretrained('../binary/tokenizer')

CHALLENGE_ID = 'challengeID'
tasks = ['gpa', 'grit', 'materialHardship', 'eviction', 'layoff', 'jobTraining']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 48


#exit(0)
def load_background(path):
  if TEST == "MEDIUM":
    df = pd.read_csv(path, skiprows=lambda x: x%100!=0)
    print("RAZOR")
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
  path = "../binary/" + label + "/final_models" 
  model = AutoModelForSequenceClassification.from_pretrained(path)
  model.eval()
  return model

def run_data(model, challenge_id, background_encoded_df, k_samples):
  df = copy.deepcopy(background_encoded_df[background_encoded_df[CHALLENGE_ID] == challenge_id])
  df = df.sample(n=k_samples)
  input_ids = torch.tensor(df['input_ids'].tolist()).to(DEVICE)
  outputs = model(input_ids)
  outputs = outputs['logits'].detach().cpu().numpy()
  ret = np.mean(outputs)
  return ret

def eval_results(predictions, golds, train_y=None, train_mean = None):
  if train_mean is None:
    train_y = train_y[~np.isnan(golds)]
    train_mean = np.mean(train_y)

  predictions = predictions[~np.isnan(golds)]
  golds = golds[~np.isnan(golds)]
  #print(len(golds))
  #print("train_mean", train_mean)
  a = np.sum(np.square(predictions - golds))
  b = np.sum(np.square(train_mean - golds))
  #print(a, b, a/b)
  R = 1 - np.sum(np.square(predictions - golds)) / np.sum(np.square(train_mean - golds))
  return R


k_df = []
task_df = []
R_df = []
train_means = {
  'gpa':2.8667381974248900,
  'grit':3.4275387870239800,
  'eviction':0.0596298834818368,
  'layoff':0.2090837901331250,
  'materialHardship':0.1037447816063310,
  'jobTraining':0.2347707049965780
}
def eval_task(main_test_df, background_encoded_df, target_label, k_samples=10):
  model = load_model(target_label).to(DEVICE)
  predictions = []
  golds = []
  for i in range(len(main_test_df)):
    current_id = main_test_df.iloc[i][CHALLENGE_ID]
    current_prediction = run_data(model, current_id, background_encoded_df, k_samples)
    predictions.append(current_prediction)
  
  final_df = copy.deepcopy(main_test_df)
  final_df['predictions'] = predictions
  results = eval_results(final_df['predictions'].to_numpy(), 
                      final_df[task].to_numpy(), 
                      train_mean = train_means[task])
  k_df.append(k_samples)
  task_df.append(target_label)
  R_df.append(results)  
  print(task, results)

max_seq_length = 512

background_df = load_background(background_path)
background_encoded_df = tokenize_background(background_df, tokenizer)
del background_df
#k_candidates = [1, 5, 10, 20, 100]
k_candidates = [10]
for k_samples in tqdm(k_candidates):
  for task in tasks:
    print("Working for", task)
    main_test_df = pd.read_csv(test_path)
    main_test_df = main_test_df[[CHALLENGE_ID, task]]
    main_test_df = main_test_df[main_test_df[task].notna()]
    eval_task(main_test_df, background_encoded_df, task, k_samples)

final_df = pd.DataFrame({"k":k_df, "task":task_df, "R":R_df})
final_df.to_csv('diff_k_normal_results.csv', index=False)
