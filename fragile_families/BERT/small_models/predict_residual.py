from transformers import RobertaTokenizerFast
from transformers import BertTokenizerFast

from transformers import AutoModelForSequenceClassification

import copy
import evaluate
import numpy as np
import os
import pandas as pd
import torch

#from mean_emb_regressor import MeanEmbRegressor
from tqdm import tqdm
import ast 

#test_small = True
test_small = False
test_path = '../../Baseline/test/'
train_path = '../../Baseline/train/'

CHALLENGE_ID = 'challengeID'
tasks = ['gpa', 'grit', 'materialHardship', 'eviction', 'layoff', 'jobTraining']

FROZEN_SINGLE_STRATEGY = "FSS"
WHOLE_SINGLE_STRATEGY = "WSS"
FROZEN_MEAN_STRATEGY = "FMS"
WHOLE_MEAN_STRATEGY = "WMS"
MODEL_STRATEGY = WHOLE_SINGLE_STRATEGY
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if MODEL_STRATEGY in [FROZEN_MEAN_STRATEGY, WHOLE_MEAN_STRATEGY]:
  BATCH_SIZE = 3
else:
  BATCH_SIZE = 48


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

def load_model(dir_path):
  path = dir_path + "/best/"
  model = AutoModelForSequenceClassification.from_pretrained(path)
  model.eval()
  return model

def run_data(model, challenge_id, background_encoded_df):
  df = copy.deepcopy(background_encoded_df[background_encoded_df[CHALLENGE_ID] == challenge_id])
  if len(df) == 0:
    print(challenge_id)
    print("FOUND NOTHING")
    assert(False)
    return 0
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
    #print(input_ids.size())
    #print("YOLO")
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



def eval_task(main_test_df, background_encoded_df, target_label, chunk_idx, dir_path, save_dict=None):
  dir_path += target_label + "_"
  if MODEL_STRATEGY == FROZEN_MEAN_STRATEGY:
    dir_path += 'frozen_mean/'
  elif MODEL_STRATEGY == WHOLE_MEAN_STRATEGY:
    dir_path += 'whole_mean/'
  elif MODEL_STRATEGY == FROZEN_SINGLE_STRATEGY:
    dir_path += 'frozen_single/'
  elif MODEL_STRATEGY == WHOLE_SINGLE_STRATEGY:
    dir_path += 'whole_single/'
  
  
  dir_path += str(chunk_idx) + "/"
  if os.path.exists(dir_path) == False:
    return

  model = load_model(dir_path).to(DEVICE)
  diff_predictions = []
  golds = []
  for i in range(len(main_test_df)):
    current_id = main_test_df.iloc[i][CHALLENGE_ID]
    current_diff_prediction = run_data(model, current_id, background_encoded_df)
    diff_predictions.append(current_diff_prediction)
    if save_dict is not None:
      save_dict[str(chunk_idx)].append(current_diff_prediction)
  final_df = copy.deepcopy(main_test_df)
  final_df['predicted_diff'] = diff_predictions
  final_df['new_prediction'] = final_df['prediction'] + final_df['predicted_diff'] 
  results = eval_results(final_df['new_prediction'].to_numpy(), 
                      final_df['truth'].to_numpy(), 
                      train_y = final_df['ybar_train'])
  print(target_label, chunk_idx, results)
  with open(dir_path+'result.txt', 'w') as f:
    f.write(str(results))
  #print(results)
  #print("*"*20 + "old" + "*"*20)
  #final_df.to_csv('results_whole/'+target_label+'_final.csv')


max_seq_length = 80

CHALLENGE_ID = 'challengeID'

tasks = ['gpa', 
      'grit', 
      'materialHardship', 
      'eviction', 
      'layoff', 
      'jobTraining']

chunk_df = pd.read_csv('../../Analysis/chunks.csv')

def predict_for_data(is_train):
  if is_train:
    path = train_path
    suffix = 'train'
  else:
    path = test_path
    suffix = 'test'
  for task in tasks:
    main_test_df = pd.read_csv(path+task+"_residual_"+suffix+".csv")
    for i in range(len(main_test_df)):
      current_id = main_test_df.iloc[i][CHALLENGE_ID]
    for i, chunk in tqdm(enumerate(chunk_df['chunk'].tolist())):
      chunk_idx = chunk_df.iloc[i]['chunk_index']
      target_dir = str(chunk_idx)+"/"
      text_path = target_dir+'text.csv'
      if os.path.exists(text_path) is False:
        continue
      tokenizer = BertTokenizerFast.from_pretrained(target_dir+'tokenizer')
      chunk = ast.literal_eval(chunk)
      background_df = pd.read_csv(text_path)
      background_encoded_df = tokenize_background(background_df, tokenizer)
      eval_task(main_test_df, background_encoded_df, task, chunk_idx, target_dir)

#predict_for_data(True)
predict_for_data(False)
