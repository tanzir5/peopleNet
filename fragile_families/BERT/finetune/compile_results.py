import pandas as pd

chunk_df = pd.read_csv('../../Analysis/chunks.csv')
BINARY_PRETRAINED = True

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
    if BINARY_PRETRAINED:
      path = "binary_"
    else:
      path = ""
    path += task+'_whole_single/'+str(chunk_idx)+'/result.txt'
    with open(path,'r') as f:
      r = float(f.read())
      tasks_for_df.append(task)
      r_for_df.append(r)
      chunk_idx_for_df.append(chunk_idx)

results_df = pd.DataFrame({'task':tasks_for_df, 'R':r_for_df, 'chunk_index':chunk_idx_for_df})
results_df = results_df.merge(chunk_df, on='chunk_index')
if BINARY_PRETRAINED:
  results_df.to_csv('binary_result_chunks.csv')
else:
  results_df.to_csv('result_chunks.csv')