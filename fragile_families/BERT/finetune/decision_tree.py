import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
import ast
test_small = True
#test_small = False
if test_small:
  LIM = 100
else:
  LIM = 1000000000
CHALLENGE_ID = 'challengeID'
test_path = '../../Baseline/test/'

def train_task(train_df, background_chunked_df, task, chunk_idx):
  x, y = get_x_y(background_chunked_df, train_df)
  regressor = DecisionTreeRegressor(random_state=0)
  regressor = regressor.fit(x, y)
  return regressor

def chunk_background_data(background_df, chunk):
  chunk.append(CHALLENGE_ID)
  ret_df = background_df[chunk]
  delete_cols = []
  for col in ret_df.columns:
    #print(ret_df[col].dtypes)
    if ret_df[col].dtypes == 'object':
      delete_cols.append(col)
    else:
      ret_df[col] = ret_df[col].fillna(ret_df[col].mean())
  ret_df = ret_df.dropna()
  ret_df.drop(delete_cols, axis='columns', inplace=True)
  print(len(ret_df))
  return ret_df

def get_x_y(background_chunked_df, df):
  new_df = df.merge(background_chunked_df, on=CHALLENGE_ID)
  y = np.array(new_df['diff'].tolist())
  new_df.drop(['diff', CHALLENGE_ID], axis='columns', inplace=True)
  x = new_df.to_numpy()
  return x, y

tasks = ['gpa', 
      'grit', 
      'materialHardship', 
      'eviction', 
      'layoff', 
      'jobTraining']

background_df = pd.read_csv('../../FFChallenge_v2/background.csv', nrows=LIM)
chunk_df = pd.read_csv('../../Analysis/chunks.csv')
for i, chunk in tqdm(enumerate(chunk_df['chunk'].tolist())):
  print("$"*100)
  print(i,"chunks done")
  chunk = ast.literal_eval(chunk)
  background_chunked_df = chunk_background_data(background_df, chunk)
  for task in tasks:
    #print("working for", task)
    train_df = pd.read_csv('../../Baseline/train/'+task+'_residual_train.csv')
    train_df = train_df[[CHALLENGE_ID, 'diff']]
    regressor = train_task(train_df, background_chunked_df, task, chunk_df.iloc[i]['chunk_index'])
    test_df = pd.read_csv(test_path+task+"_residual_test.csv")
    test_df = test_df[[CHALLENGE_ID, 'diff']]
    #print(test_df.columns)
    test_x, test_y = get_x_y(background_chunked_df, test_df)
    if(len(test_x) > 0):
      pred = regressor.predict(test_x)
      print(np.mean(np.abs(test_y - pred)))
    #print(np.mean(np.abs(test_y)))
    #input()