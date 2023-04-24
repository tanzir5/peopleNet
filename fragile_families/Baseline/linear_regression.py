import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

tasks = ['gpa', 'grit', 'materialHardship', 'eviction', 'layoff', 'jobTraining']

def get_xy(df, label):
  train_cols = list(df.columns)
  train_cols.remove(label)
  if 'Unnamed: 0' in train_cols:
    train_cols.remove('Unnamed: 0')
  x = df[train_cols].to_numpy()
  y = df[label].to_numpy()
  return x, y

def do_linear_regression(train_df, label):
  x, y = get_xy(train_df, label)
  reg = LinearRegression().fit(x, y)
  print("reg score:", reg.score(x,y))
  return reg

def get_stats(x):
  return np.min(x), np.median(x), np.max(x), np.mean(x), np.std(x)

def get_prediction(reg, test_df, label):
  x, y = get_xy(test_df, label)
  preds = reg.predict(x)
  print("preds stats:", get_stats(preds))
  return preds, y

def eval_results(predictions, golds, train_y):
  train_mean = np.mean(train_y)
  print("train_mean", train_mean)
  a = np.sum(np.square(predictions - golds))
  b = np.sum(np.square(train_mean - golds))
  print(a, b, a/b)
  R = 1 - np.sum(np.square(predictions - golds)) / np.sum(np.square(train_mean - golds))
  return R

for task in tasks:
  train_df = pd.read_csv('train/'+task+'_train.csv')
  test_df = pd.read_csv('test/'+task+'_leaderboard.csv')
  #test_df = pd.read_csv('test/'+task+'_test.csv')
  reg = do_linear_regression(train_df, task)
  preds, golds = get_prediction(reg, test_df, task)
  _, train_y = get_xy(train_df, task)
  print("training data:", len(train_y))
  print("test data:", len(preds))
  print("r2 holdout results:",eval_results(preds, golds, train_y))
  break