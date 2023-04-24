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
test_path = '../../Baseline/test/'
train_path = '../train.csv'

CHALLENGE_ID = 'challengeID'
tasks = ['gpa', 'grit', 'materialHardship', 'eviction', 'layoff', 'jobTraining']


BATCH_SIZE = 48
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
  path = label + "_residual_frozen/best/"
  model = AutoModelForSequenceClassification.from_pretrained(path)
  return model

def run_data(model, challenge_id, background_encoded_df):
  #print("Hello", challenge_id)
  #print(background_encoded_df[CHALLENGE_ID])
  df = copy.deepcopy(background_encoded_df[background_encoded_df[CHALLENGE_ID] == challenge_id])
  input_ids = torch.tensor(df['input_ids'].tolist()).to(DEVICE)
  outputs = model(input_ids)
  #print(outputs)
  outputs = outputs['logits'].detach().cpu().numpy()
  #print("output shape", outputs.shape)
  return outputs.flatten()

def eval_results(predictions, golds, train_y=None, train_mean = None):
  if train_mean is None:
    train_y = train_y[~np.isnan(golds)]
    train_mean = np.mean(train_y)

  predictions = predictions[~np.isnan(golds)]
  golds = golds[~np.isnan(golds)]
  print(len(golds))
  print("train_mean", train_mean)
  a = np.sum(np.square(predictions - golds))
  b = np.sum(np.square(train_mean - golds))
  print(a, b, a/b)
  R = 1 - np.sum(np.square(predictions - golds)) / np.sum(np.square(train_mean - golds))
  return R

def find_nearest(array, value):
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return array[idx]

def eval_task(main_test_df, background_encoded_df, target_label):
  model = load_model(target_label).to(DEVICE)
  diff_predictions = []
  golds = []
  for i in range(len(main_test_df)):
    current_id = main_test_df.iloc[i][CHALLENGE_ID]
    current_diff_prediction = run_data(model, current_id, background_encoded_df)
    diff_predictions.append(current_diff_prediction)
  
  final_df = copy.deepcopy(main_test_df)
  final_df['predicted_diff_list'] = diff_predictions
  predicted_diff_best = []
  for i, diffs in enumerate(final_df['predicted_diff_list']):
    real_diff = final_df.iloc[i]['diff']
    predicted_diff_best.append(find_nearest(diffs, real_diff))

  final_df['predicted_diff_best'] = predicted_diff_best
  final_df['unfair_prediction'] = final_df['prediction'] + final_df['predicted_diff_best'] 
  
  results = eval_results(final_df['unfair_prediction'].to_numpy(), 
                      final_df['truth'].to_numpy(), 
                      train_y = final_df['ybar_train'])
  print(results)
  print("*"*20 + "old" + "*"*20)
  results = eval_results(final_df['prediction'].to_numpy(), 
                      final_df['truth'].to_numpy(), 
                      train_y = final_df['ybar_train'])

  print(results)
  #final_df.to_csv('results_whole/'+target_label+'_final.csv')

max_seq_length = 512
tokenizer = RobertaTokenizerFast(tokenizer_file="../byte-level-BPE.tokenizer.json", max_len=max_seq_length)


background_df = load_background(background_path)
background_encoded_df = tokenize_background(background_df, tokenizer)

for task in tasks:
  print("Working for", task)
  main_test_df = pd.read_csv(test_path+task+"_residual_test.csv")
  eval_task(main_test_df, background_encoded_df, task)
