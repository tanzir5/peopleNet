import pandas as pd
import numpy as np

tasks = [
      "A. Material\nhardship",
      "B. GPA",
      "C. Grit",
      "D. Eviction",
      "E. Job\ntraining",
      "F. Layoff",]

df = pd.read_csv('benchmarks_long.csv')
print(len(df))
print(df.columns)


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



for i, task in enumerate(tasks):
  print("working for", task)
  account = 'benchmark_ols_full' if i < 3 else 'benchmark_logit_full'
  task_df = df[(df['outcome_name']==task) & (df['account'] == account)]
  print("R2 value:",round(eval_results(task_df['prediction'].to_numpy(), task_df['truth'].to_numpy(), train_y=task_df['ybar_train'].to_numpy()),2))