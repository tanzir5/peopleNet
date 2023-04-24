import pandas as pd
import copy
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
test_path = '../../Baseline/test/'
train_path = '../../Baseline/train/'

pred_train = pd.read_csv('chunk_preds_train.csv')
pred_test = pd.read_csv('chunk_preds_test.csv')
CHALLENGE_ID = 'challengeID'

tasks = ['gpa', 
      'grit', 
      'materialHardship', 
      'eviction', 
      'layoff', 
      'jobTraining']

def get_stats(a):
  return np.min(a), np.median(a), np.max(a), np.mean(a), np.std(a)


def load_df(is_train, task):
  if is_train:
    main_df = pd.read_csv(train_path+task+"_residual_train.csv")
    task_preds = copy.deepcopy(pred_train[pred_train['task']==task])
  else:
    main_df = pd.read_csv(test_path+task+"_residual_test.csv")
    task_preds = copy.deepcopy(pred_test[pred_test['task']==task])
  main_df = main_df[[CHALLENGE_ID, 'diff']]  
  task_preds = task_preds.merge(main_df, on=CHALLENGE_ID)
  y = task_preds['diff'].to_numpy()
  task_preds.drop([CHALLENGE_ID, 'diff', 'task'], axis='columns', inplace=True)
  x = task_preds.to_numpy()
  return x, y

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

def eval_results_df(diff_predictions, df):
  final_df = copy.deepcopy(df)
  final_df['predicted_diff'] = diff_predictions
  final_df['new_prediction'] = final_df['prediction'] + final_df['predicted_diff'] 
  results = eval_results(final_df['new_prediction'].to_numpy(), 
                      final_df['truth'].to_numpy(), 
                      train_y = final_df['ybar_train'])
  return results

def booster_solve(train_x, train_y, test_x, test_y, task):
  reg = AdaBoostRegressor(random_state=5, 
    loss='linear', 
    learning_rate=1e-2,
    ) 
    #n_estimators=10)
  reg = reg.fit(train_x, train_y)
  train_preds = reg.predict(train_x)
  test_preds = reg.predict(test_x)
  bla = pd.DataFrame({'p':test_preds, 'g':test_y})
  #print(bla.corr())
  #exit(0)
  #print(get_stats(train_preds))
  #print(get_stats(test_preds))
  #print(np.mean(np.abs(train_preds)))
  #print(np.mean(np.abs(test_preds)))
  #print("train error", np.mean(np.abs(train_preds-train_y)))
  #print("test error", np.mean(np.abs(test_preds-test_y)))
  
  train_df = pd.read_csv(train_path+task+"_residual_train.csv")
  test_df = pd.read_csv(test_path+task+"_residual_test.csv")

  train_R = eval_results_df(train_preds, train_df)
  test_R = eval_results_df(test_preds, test_df)
  gold_R = eval_results_df(np.zeros(len(train_df)), train_df)
  reported_test_R = eval_results_df(np.zeros(len(test_df)), test_df)
  print("Train & Test Rs:", train_R, test_R, gold_R, reported_test_R)
  print(test_R - reported_test_R)

for task in tasks:
  print("working for ", task)
  train_x, train_y = load_df(True, task)
  test_x, test_y = load_df(False, task)
  #print(train_x)
  booster_solve(train_x, train_y, test_x, test_y, task)
  #exit(0)