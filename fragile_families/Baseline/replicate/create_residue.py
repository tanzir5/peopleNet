import pandas as pd
import numpy as np

tasks = [
      "A. Material\nhardship",
      "B. GPA",
      "C. Grit",
      "D. Eviction",
      "E. Job\ntraining",
      "F. Layoff",]

df = pd.read_csv('benchmarks_long_test.csv')
print(len(df))
print(df.columns)

def get_stats(a):
  return np.min(a), np.median(a), np.max(a), np.mean(a), np.std(a)

def eval_results(predictions, golds, train_y=None, train_mean = None):
  if train_mean is None:
    train_y = train_y[~np.isnan(golds)]
    train_mean = np.mean(train_y)

  predictions = predictions[~np.isnan(golds)]
  golds = golds[~np.isnan(golds)]
  R = 1 - np.sum(np.square(predictions - golds)) / np.sum(np.square(train_mean - golds))
  return R

req_columns = ['challengeID', 'prediction', 'truth', 'diff', 'outcome', 'account', 'ybar_train']

for i, task in enumerate(tasks):
  print("working for", task)
  account = 'benchmark_ols_full' if i < 3 else 'benchmark_logit_full'
  task_df = df[(df['outcome_name']==task) & (df['account'] == account)]
  
  task_df = task_df[task_df['truth'].notnull()]
  task_df['diff'] = task_df['truth'] - task_df['prediction']
  
  print("R2 value:",round(eval_results(task_df['prediction'].to_numpy(), task_df['truth'].to_numpy(), train_y=task_df['ybar_train'].to_numpy()),5))
  print(get_stats(task_df['diff'].to_numpy()))
  task_df = task_df[req_columns]
  task_df.to_csv('../test/'+task_df.iloc[0]['outcome']+'_residual_test.csv', index=False)