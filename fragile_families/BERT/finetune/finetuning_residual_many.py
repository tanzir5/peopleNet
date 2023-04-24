import ast
import copy
import evaluate
import numpy as np
import os
import pandas as pd
import torch.nn as nn
import torch

from datasets import Dataset
from datasets import load_dataset
from transformers import AdamW
from transformers import RobertaTokenizerFast
from transformers import BertTokenizerFast
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import AutoModel
from transformers import EarlyStoppingCallback
from tqdm import tqdm

from torch.utils.data import DataLoader

BINARY_PRETRAINED = True

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
EPOCHS = 20
max_seq_length = 512

#test_small = True
test_small = False
if test_small:
  LIM = 100
else:
  LIM = 1000000000


if BINARY_PRETRAINED is False:
  background_path = '../../FFChallenge_v2/background.csv'
  tokenizer = RobertaTokenizerFast(tokenizer_file="../byte-level-BPE.tokenizer.json", max_len=max_seq_length)
  model_path = '../final_models'
else:
  background_path = '../../Analysis/binary_background.csv'
  tokenizer = BertTokenizerFast.from_pretrained('../binary/tokenizer')
  model_path = '../binary/final_models'

def compute_classification_metrics(eval_pred):
  metric = evaluate.load("accuracy")
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels)

def get_stats(a):
  return np.min(a), np.median(a), np.max(a), np.mean(a), np.std(a)


def compute_regression_metrics(eval_pred):
  metric = evaluate.load("mae")
  predictions, labels = eval_pred
  print("stats:",get_stats(predictions))
  print("gold stats", get_stats(labels))
  #R = np.sum(np.square(labels - predictions)) / np.sum(np.square(labels)) 
  #print("R2 ratio:", R)
  #print(get_stats(np.absolute(labels-predictions)/np.absolute(labels)))
  return metric.compute(predictions=predictions, references=labels)

def get_train_dict(main_train_df, background_encoded_df):
  idxs = []
  input_ids = []
  labels = []
  for i in range(len(main_train_df)):
    current_idx = main_train_df.iloc[i][CHALLENGE_ID]
    current_idx_rows = background_encoded_df[background_encoded_df[CHALLENGE_ID] == current_idx]
    if (background_encoded_df[CHALLENGE_ID] == current_idx).any() is False:
      continue
    current_input_ids = current_idx_rows['input_ids'].tolist()
    label = main_train_df.iloc[i]['diff']
    current_labels = [label] * len(current_idx_rows)
    labels.extend(current_labels)
    input_ids.extend(current_input_ids)
    idxs.extend([current_idx]* len(current_idx_rows))
  return {
    CHALLENGE_ID: idxs, 
    'input_ids':input_ids, 
    'label':labels,
    }

class MeanDataset(Dataset):
  def __init__(self, dict):
    dict[CHALLENGE_ID] = np.array(dict[CHALLENGE_ID])
    dict['input_ids'] = np.array(dict['input_ids'])
    dict['label'] = np.array(dict['label'])
    self.dict = dict
    self.ids = list(set(self.dict[CHALLENGE_ID]))
    self.length = len(self.ids)
    '''
    ids = set(dict[CHALLENGE_ID])
    input_ids = []
    labels = []
    for i in ids:
      input_ids.append(dict['input_ids'][dict[CHALLENGE_ID] == i])
      labels.append()
    '''
  def __getitem__(self, index):
    target_id = self.ids[index]
    x = (self.dict['input_ids'][self.dict[CHALLENGE_ID] == target_id])
    y = (self.dict['label'][self.dict[CHALLENGE_ID] == target_id])[0]
    #print(x.shape)
    #print(y.shape)

    return x, y

  def __len__(self):
    #print("YO")
    #print(self.length)
    return self.length

def get_train_eval_dataset(main_train_df, background_encoded_df, task, doing_mean_emb=False):
  train_dict = get_train_dict(main_train_df, background_encoded_df)
  if doing_mean_emb == False:
    train_ds = Dataset.from_dict(train_dict)
    train_ds = train_ds.shuffle(seed=42)
    ds = train_ds.train_test_split(test_size=0.1)
    train_ds = ds['train']
    eval_ds = ds['test']
  else:
    train_ds = MeanDataset(train_dict)
    eval_ds = None
  
  return train_ds, eval_ds

def single_embedding_training(model, output_dir, train_ds, eval_ds):
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
       compute_metrics=compute_regression_metrics,
       callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
   )
  #print(train_ds)
  #print(eval_ds)
  trainer.train()
  final_path = output_dir+"/best"
  if os.path.exists(final_path) == False:
    os.mkdir(final_path) 
  trainer.save_model(final_path)

def eval(model, data_loader):
  model.eval()
  total_loss = 0

  with torch.no_grad():
    for (X,Y) in tqdm(data_loader):
      X = X.to(DEVICE)
      Y = Y.to(DEVICE).float()
      result = model(X,Y)
      
      loss = result['loss']
      output = result['out']
      total_loss += loss.item()
  
  return total_loss/len(data_loader)


def mean_model_train_save(model, main_train_ds, optimizer, save_path):
  train_ds = torch.utils.data.Subset(main_train_ds, range(0, int(np.ceil(len(main_train_ds)*0.9))))
  eval_ds = torch.utils.data.Subset(main_train_ds, range(int(np.ceil(len(main_train_ds)*0.9)), len(main_train_ds)))
  train_dataloader = DataLoader(train_ds,shuffle=True,batch_size=BATCH_SIZE)
  eval_dataloader = DataLoader(eval_ds,shuffle=True,batch_size=BATCH_SIZE)
  model.train()
  #print("voila")
  best_val_loss = 1e9
  consecutive_val_loss_increase_count = 0
  patience = 3
  for epoch in range(EPOCHS):
    total_loss = 0
    for batch_idx, (X,Y) in enumerate(tqdm(train_dataloader)):
      #print("!"*100)
      model.zero_grad()
      X = X.to(DEVICE)
      Y = Y.to(DEVICE).float()
      
      result = model(X,Y)
      
      loss = result['loss']
      output = result['out']
      loss.backward()
      optimizer.step()
              
      #precision, recall, f1 = get_performance_metrics(output, labels)

      total_loss += loss.item()

    total_loss /= len(train_dataloader)
    total_val_loss = eval(model, eval_dataloader)
    print("Epoch:", epoch, "loss:", total_loss, "val_loss:", total_val_loss)
    if total_val_loss < best_val_loss:
      best_val_loss = total_val_loss
      torch.save(model, save_path)
      consecutive_val_loss_increase_count = 0
    else:
      consecutive_val_loss_increase_count += 1
      if consecutive_val_loss_increase_count >= patience:
        break

  return model

class MeanEmbRegressor(nn.Module):
  def __init__(self, emb_model):
    super(MeanEmbRegressor, self).__init__()
    self.emb_model = emb_model.to(DEVICE)
    self.regressor = nn.Linear(60, 1)
    self.loss_fn = nn.MSELoss()

  def forward(self,x,y):
    y = torch.reshape(y, (-1, 1))
    #print(y.size())
    batch_embs = None
    for input_ids in x: 
      samples_tokens_embs = self.emb_model(input_ids)[0]
      #print(samples_tokens_embs.size())
      samples_embs = torch.mean(samples_tokens_embs, axis = 1)
      #print(samples_embs.size())
      emb = torch.mean(samples_embs, axis = 0)
      emb = torch.reshape(emb, (1, -1))
      #print(emb.size())
      if batch_embs is None:
        batch_embs = emb
      else:
        batch_embs = torch.cat((batch_embs, emb))
      #print(batch_embs.size())
    out = self.regressor(batch_embs)
    #print(out.shape)
    self.loss = self.loss_fn(out,y)
    #print(self.loss)
    return {'out':out, 'loss':self.loss}


def mean_embedding_training(pretrained_model, output_dir, train_ds, eval_ds):
  model = MeanEmbRegressor(pretrained_model).to(DEVICE)
  print(model)
  if MODEL_STRATEGY == WHOLE_MEAN_STRATEGY:
    lr = 5e-05
  else:
    lr = 5e-05
  optimizer = AdamW(model.parameters(), lr = lr, eps = 1e-8)
  if os.path.exists(output_dir) is False:
    os.mkdir(output_dir)
  save_path = output_dir+'/best_model.pt'
  model = mean_model_train_save(model, train_ds, optimizer, save_path)

def freeze_pretrained_layers(model):
  for name, param in model.named_parameters():
    #print(param)
    if 'classifier' not in name: # classifier layer
      param.requires_grad = False
  return model 

def train_task(main_train_df, background_encoded_df, task, chunk_idx):
  train_ds, eval_ds = get_train_eval_dataset(main_train_df, background_encoded_df, task)
  #print(type(train_ds))
  #print(len(train_ds))
  #print(train_ds)
  num_labels = 1
  compute_metrics = compute_regression_metrics
  
  if BINARY_PRETRAINED:
    output_dir = "binary_"
  else:
    output_dir = ""
  
  output_dir += task + "_"
  if MODEL_STRATEGY == FROZEN_MEAN_STRATEGY:
    output_dir += 'frozen_mean/'
  elif MODEL_STRATEGY == WHOLE_MEAN_STRATEGY:
    output_dir += 'whole_mean/'
  elif MODEL_STRATEGY == FROZEN_SINGLE_STRATEGY:
    output_dir += 'frozen_single/'
  elif MODEL_STRATEGY == WHOLE_SINGLE_STRATEGY:
    output_dir += 'whole_single/'
  
  if os.path.exists(output_dir) == False:
    os.mkdir(output_dir)

  output_dir += str(chunk_idx) + "/"
  
  if MODEL_STRATEGY in [FROZEN_MEAN_STRATEGY, WHOLE_MEAN_STRATEGY]:
    model = AutoModel.from_pretrained(model_path)
    if MODEL_STRATEGY == FROZEN_MEAN_STRATEGY:
      model = freeze_pretrained_layers(model)
    mean_embedding_training(model, output_dir, train_ds, eval_ds)
  else:
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    if MODEL_STRATEGY == FROZEN_SINGLE_STRATEGY:
      model = freeze_pretrained_layers(model)
    single_embedding_training(model, output_dir, train_ds, eval_ds)  
  #exit(0)
  
def load_background(path):
  if TEST != "ALL":
    df = pd.read_csv(path, skiprows=lambda x: x%100!=0)
    print("small/medium loaded, columns=", df.columns)
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

def create_encode_background_data(background_df, cols, tokenizer):
  challenge_ids = []
  texts = []
  for i in range(len(background_df)):
    text = ""
    for col in cols:
      if col in background_df.columns:
        val = background_df[col].iloc[i]
        if pd.isna(val):
          continue
        text += col
        text += " "
        text += str(int(val))
        text += " , "
    if text == "":
      continue
    texts.append(text)
    challenge_ids.append(background_df[CHALLENGE_ID].iloc[i])
  if len(texts) < 100:
    return None
  df = pd.DataFrame({CHALLENGE_ID:challenge_ids, 'text': texts})
  encoded_df = tokenize_background(df, tokenizer)
  return encoded_df

CHALLENGE_ID = 'challengeID'
tasks_type = {
  "gpa":"regression",
  "grit":"regression",
  "materialHardship":"regression",
  "eviction":"binary",
  "layoff":"binary",
  "jobTraining":"binary"
}
tasks = ['gpa', 
      'grit', 
      'materialHardship', 
      'eviction', 
      'layoff', 
      'jobTraining']

background_df = pd.read_csv(background_path, nrows=LIM)
chunk_df = pd.read_csv('../../Analysis/chunks.csv')

for i, chunk in tqdm(enumerate(chunk_df['chunk'].tolist())):
  print("$"*100)
  print(i,"chunks done")
  chunk = ast.literal_eval(chunk)
  background_encoded_df = create_encode_background_data(background_df, chunk, tokenizer)
  if background_encoded_df is None:
    continue
  for task in tasks:
    print("working for", task)
    train_df = pd.read_csv('../../Baseline/train/'+task+'_residual_train.csv')
    train_task(train_df, background_encoded_df, task, chunk_df.iloc[i]['chunk_index'])
