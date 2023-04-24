from transformers import RobertaTokenizerFast
from transformers import AutoModelForSequenceClassification

import copy
import evaluate
import numpy as np
import os
import pandas as pd
import torch

from mean_emb_regressor import MeanEmbRegressor
from tqdm import tqdm
import ast 

BINARY_PRETRAINED = True

if BINARY_PRETRAINED is False:
  background_path = '../../FFChallenge_v2/background.csv'
  tokenizer = RobertaTokenizerFast(tokenizer_file="../byte-level-BPE.tokenizer.json", max_len=max_seq_length)
else:
  background_path = '../../Analysis/binary_background.csv'
  tokenizer = BertTokenizerFast.from_pretrained('../binary/tokenizer')

#test_small = True
test_small = False
if test_small:
  LIM = 100
else:
  LIM = 1000000000
test_path = '../../Baseline/test/'
train_path = '../../Baseline/train/'

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



def eval_task(main_test_df, background_encoded_df, target_label, chunk_idx, save_dict=None):
  dir_path = str(target_label)+"_residual_frozen_single/"+str(chunk_idx)+"/"
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
  with open(dir_path+'result.txt', 'w') as f:
    f.write(str(results))
  #print(results)
  #print("*"*20 + "old" + "*"*20)
  #final_df.to_csv('results_whole/'+target_label+'_final.csv')


def create_encode_background_data(background_df, cols, tokenizer):
  challenge_ids = []
  texts = []
  for i in range(len(background_df)):
    text = ""
    for col in cols:
      text += col
      text += " "
      text += str(background_df[col].iloc[i])
      text += " , "
    texts.append(text)
    challenge_ids.append(background_df[CHALLENGE_ID].iloc[i])
  df = pd.DataFrame({CHALLENGE_ID:challenge_ids, 'text': texts})
  encoded_df = tokenize_background(df, tokenizer)
  return encoded_df


max_seq_length = 512
tokenizer = RobertaTokenizerFast(tokenizer_file="../byte-level-BPE.tokenizer.json", max_len=max_seq_length)

CHALLENGE_ID = 'challengeID'
tasks_type = {
  "gpa":"regression",
  "grit":"regression",
  "materialHardship":"regression",
  "eviction":"binary",
  "layoff":"binary",
  "jobTraining":"binary"
}
tokenizer = RobertaTokenizerFast(tokenizer_file="../byte-level-BPE.tokenizer.json", max_len=max_seq_length)
tasks = ['gpa', 
      'grit', 
      'materialHardship', 
      'eviction', 
      'layoff', 
      'jobTraining']

background_df = pd.read_csv('../../FFChallenge_v2/background.csv', nrows=LIM)
chunk_df = pd.read_csv('../../Analysis/chunks.csv')

def predict_for_data(is_train):
  save_dict = {}
  save_dict[CHALLENGE_ID] = []
  save_dict['task'] = []

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
      save_dict[CHALLENGE_ID].append(current_id)
      save_dict['task'].append(task)
    for i, chunk in tqdm(enumerate(chunk_df['chunk'].tolist())):
      chunk_idx = chunk_df.iloc[i]['chunk_index']
      if str(chunk_idx) not in save_dict:
        save_dict[str(chunk_idx)] = []
      chunk = ast.literal_eval(chunk)
      background_encoded_df = create_encode_background_data(background_df, chunk, tokenizer)
      eval_task(main_test_df, background_encoded_df, task, chunk_idx, save_dict)

  pred_df = pd.DataFrame.from_dict(save_dict)
  pred_df.to_csv('chunk_preds_'+suffix+'.csv',index=False)

predict_for_data(True)
predict_for_data(False)
'''
for i, chunk in tqdm(enumerate(chunk_df['chunk'].tolist())):
  chunk_idx = chunk_df.iloc[i]['chunk_index']
  save_dict[chunk_idx] = []
  print("$"*100)
  print(i,"chunks done")
  chunk = ast.literal_eval(chunk)
  background_encoded_df = create_encode_background_data(background_df, chunk, tokenizer)
  for task in tasks:
    #print("working for", task)
    main_test_df = pd.read_csv(test_path+task+"_residual_test.csv")
    eval_task(main_test_df, background_encoded_df, task, chunk_df.iloc[i]['chunk_index'])
'''