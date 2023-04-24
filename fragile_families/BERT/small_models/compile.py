import pandas as pd
import os

chunk_df = pd.read_csv('../../Analysis/chunks.csv')
tasks = ['gpa', 
      'grit', 
      'materialHardship', 
      'eviction', 
      'layoff', 
      'jobTraining']

tasks_for_df = []
r_for_df = []
chunk_idx_for_df = []
for task in tasks:
  for chunk_idx in range(67):
    path = str(chunk_idx)+"/"+task+'_whole_single/'+str(chunk_idx)+'/result.txt'
    if os.path.exists(path) is False:
        continue
    with open(path,'r') as f:
      r = float(f.read())
      tasks_for_df.append(task)
      r_for_df.append(r)
      chunk_idx_for_df.append(chunk_idx)

results_df = pd.DataFrame({'task':tasks_for_df, 'R':r_for_df, 'chunk_index':chunk_idx_for_df})
results_df = results_df.merge(chunk_df, on='chunk_index')
results_df.to_csv('result_chunks.csv')