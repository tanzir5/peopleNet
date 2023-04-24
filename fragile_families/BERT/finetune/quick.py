import pandas as pd

df = pd.read_csv('binary_diff_k_results.csv')

tasks = ['eviction',
        'gpa',
        'grit',
        'jobTraining',
        'layoff',
        'materialHardship']

dic = {'1':[], '5':[], '10':[], '20':[], '100':[]}
for task in tasks:
  for k in [1, 5, 10, 20, 100]:
    val = df[(df['task'] == task) & (df['k'] == k)]['R'].iloc[0]
    dic[str(k)].append(val)

df = pd.DataFrame.from_dict(dic)
df.to_csv('new_k.csv')