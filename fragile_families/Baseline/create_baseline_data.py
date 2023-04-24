import copy
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

#test_small = True
test_small = False
if test_small:
  LIM = 100
else:
  LIM = 1000000000
tasks = ['gpa', 'grit', 'materialHardship', 'eviction', 'layoff', 'jobTraining']
common_features = ['cm1ethrace', 'cm1edu', 'cm1relf']

def impute_missing_values(data, strategy):
  data = np.array(data, dtype='float32')
  data[data < 0] = np.nan
  imputer = SimpleImputer(strategy=strategy)
  data = np.reshape(data, (len(data), 1))
  imputed_data = imputer.fit_transform(data)
  return imputed_data

def construct_gpa_proxy_feature(df, cols):
  relevant_cols = ['t5c13a', 't5c13b', 't5c13c']
  gpa = None
  for col in relevant_cols:
    data = impute_missing_values(df[col], 'mean').squeeze()
    df[col] = data
    if gpa is None:
      gpa = copy.deepcopy(data)
    else:
      gpa += data
  gpa /= 3
  df['gpa_proxy'] = gpa
  cols.append('gpa_proxy')
  return df, cols

def construct_grit_proxy_feature(df, cols):
  pass

def construct_proxy_feature(df, task, cols):
  if task == 'gpa':
    return construct_gpa_proxy_feature(df, cols)
  elif task == 'grit':
    return construct_grit_proxy_feature(df, cols)
  elif task == 'materialHardship':
    return construct_material_hardship_proxy_feature(df, cols)
  elif task == 'eviction':
    return construct_eviction_proxy_feature(df, cols)
  elif task == 'layoff':
    return construct_layoff_proxy_feature(df, cols)
  elif task == 'jobTraining':
    return construct_job_training_proxy_feature(df, cols)

def construct_race_feature(df):
  df['cm1ethrace'] = impute_missing_values(df['cm1ethrace'], 'most_frequent')
  
  race_black = np.zeros(len(df))
  race_black |= df['cm1ethrace'] == 2
  df['race_black'] = race_black
  
  race_hispanic = np.zeros(len(df))
  race_hispanic |= df['cm1ethrace'] == 3
  df['race_hispanic'] = race_hispanic

  race_others = np.zeros(len(df))
  race_others |= ((df['cm1ethrace'] != 2) & (df['cm1ethrace'] != 3))
  df['race_others'] = race_others

  return df

def construct_edu_feature(df):
  df['cm1edu'] = impute_missing_values(df['cm1edu'], 'mean')
  return df

def construct_marital_feature(df):
  df['cm1relf'] = impute_missing_values(df['cm1relf'], 'most_frequent')
  
  marital_married = np.zeros(len(df))
  marital_married |= df['cm1relf'] == 1
  df['marital_married'] = marital_married
  
  marital_cohab = np.zeros(len(df))
  marital_cohab |= df['cm1relf'] == 2
  df['marital_cohab'] = marital_cohab

  marital_others = np.zeros(len(df))
  marital_others |= ((df['cm1relf'] != 1) & (df['cm1relf'] != 2))
  df['marital_others'] = marital_others

  return df

def construct_common_features(df):
  df = construct_race_feature(df)
  df = construct_edu_feature(df)
  df = construct_marital_feature(df)
  return df



def train_test_split(task_df, train_df, test_df, task_name):
  new_train_df = copy.deepcopy(train_df)
  new_train_df = new_train_df[['challengeID', task_name]]
  new_train_df.drop(new_train_df.loc[new_train_df[task_name].isnull()].index, inplace=True)

  new_test_df = copy.deepcopy(test_df)
  new_test_df = new_test_df[['challengeID', task_name]]
  new_test_df.drop(new_test_df.loc[new_test_df[task_name].isnull()].index, inplace=True)
  
  new_train_df = new_train_df.merge(task_df, on='challengeID')
  new_test_df = new_test_df.merge(task_df, on='challengeID')
  return new_train_df, new_test_df

df = pd.read_csv('../FFChallenge_v2/background.csv', nrows=LIM)
#test_df = pd.read_csv('../FFChallenge_v2/leaderboard.csv')
train_df = pd.read_csv('../FFChallenge_v2/train.csv')
test_df = pd.read_csv('../FFChallenge_v2/test.csv')
leaderboard_df = pd.read_csv('../FFChallenge_v2/leaderboard.csv')

df = construct_common_features(df)
required_cols = [ 'challengeID',
                  'race_black', 'race_hispanic', 'race_others', 
                  'cm1edu', 
                  'marital_married', 'marital_cohab', 'marital_others']
for task in tasks:
  print("Task: ", task)
  task_required_cols = copy.deepcopy(required_cols)
  task_df = copy.deepcopy(df)
  task_df, task_required_cols = construct_proxy_feature(task_df, task, task_required_cols)
  task_df = task_df[task_required_cols]
  train_task_df, test_task_df = train_test_split(task_df, train_df, test_df, task)
  print(task_df.columns)
  print(task_df.describe())
  train_task_df.to_csv('train/'+task+'_train.csv', index='challengeID')
  test_task_df.to_csv('test/'+task+'_test.csv', index='challengeID')
  train_task_df, leaderboard_task_df = train_test_split(task_df, train_df, leaderboard_df, task)
  leaderboard_task_df.to_csv('test/'+task+'_leaderboard.csv', index='challengeID')
  break
