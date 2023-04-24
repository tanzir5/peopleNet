from transformers import RobertaTokenizerFast
from transformers import AutoModelForSequenceClassification

import copy
import evaluate
import numpy as np
import os
import pandas as pd
import torch

from mean_emb_regressor import MeanEmbRegressor

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


FROZEN_SINGLE_STRATEGY = "FSS"
WHOLE_SINGLE_STRATEGY = "WSS"
FROZEN_MEAN_STRATEGY = "FMS"
WHOLE_MEAN_STRATEGY = "WMS"
MODEL_STRATEGY = FROZEN_SINGLE_STRATEGY
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if MODEL_STRATEGY in [FROZEN_MEAN_STRATEGY, WHOLE_MEAN_STRATEGY]:
  BATCH_SIZE = 3
else:
  BATCH_SIZE = 48
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
  if MODEL_STRATEGY == WHOLE_MEAN_STRATEGY:
    path = label + '_residual_whole_mean/best_model.pt'
    model = torch.load(path, map_location=DEVICE)
  elif MODEL_STRATEGY == FROZEN_MEAN_STRATEGY:
    path = label + '_residual_frozen_mean/best_model.pt'
    model = torch.load(path, map_location=DEVICE)  
  elif MODEL_STRATEGY == FROZEN_SINGLE_STRATEGY:
    path = label + "_residual_frozen/best/"
    model = AutoModelForSequenceClassification.from_pretrained(path)
  else:
    assert(False)
  model.eval()
  return model

def run_data(model, challenge_id, background_encoded_df):
  df = copy.deepcopy(background_encoded_df[background_encoded_df[CHALLENGE_ID] == challenge_id])
  if MODEL_STRATEGY in [FROZEN_MEAN_STRATEGY, WHOLE_MEAN_STRATEGY]:
    input_ids = torch.tensor(df['input_ids'].tolist()).to(DEVICE)
    dummy_label = torch.tensor(2).to(DEVICE)
    input_ids = torch.reshape(input_ids, (1, -1, 512))
    dummy_label = torch.reshape(dummy_label, (1, -1))
    #print("input_ids", input_ids.size())
    #print(dummy_label.size())
    output = model(input_ids, dummy_label)['out']
    ret = torch.squeeze(output).cpu().detach().numpy()
  else:  
    #df = df.sample(n=10)
    input_ids = torch.tensor(df['input_ids'].tolist()).to(DEVICE)
    outputs = model(input_ids)
    outputs = outputs['logits'].detach().cpu().numpy()
    ret1 = np.mean(outputs)
    ret2 = np.median(outputs)
    return ret1, ret2
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



def eval_task(main_test_df, background_encoded_df, target_label):
  model = load_model(target_label).to(DEVICE)
  diff_predictions1 = []
  diff_predictions2 = []
  golds = []
  for i in range(len(main_test_df)):
    current_id = main_test_df.iloc[i][CHALLENGE_ID]
    current_diff_prediction1, current_diff_prediction2 = run_data(model, current_id, background_encoded_df)
    diff_predictions1.append(current_diff_prediction1)
    diff_predictions2.append(current_diff_prediction2)
  final_df = copy.deepcopy(main_test_df)
  final_df['predicted_diff1'] = diff_predictions1
  final_df['predicted_diff2'] = diff_predictions2
  final_df['new_prediction1'] = final_df['prediction'] + final_df['predicted_diff1'] 
  results = eval_results(final_df['new_prediction1'].to_numpy(), 
                      final_df['truth'].to_numpy(), 
                      train_y = final_df['ybar_train'])
  print(results)
  final_df['new_prediction2'] = final_df['prediction'] + final_df['predicted_diff2'] 
  results = eval_results(final_df['new_prediction2'].to_numpy(), 
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
