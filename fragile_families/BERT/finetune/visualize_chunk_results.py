import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
st = sns.axes_style("dark")
BINARY_PRETRAINED = True

if BINARY_PRETRAINED:
  df = pd.read_csv('binary_result_chunks.csv')
else:
  df = pd.read_csv('result_chunks.csv')
tasks = ['gpa', 
      'grit', 
      'materialHardship', 
      'eviction', 
      'layoff', 
      'jobTraining']

four_factors = { 
      'gpa':0.1059, 
      'grit':0.0146, 
      'materialHardship':0.1773, 
      'eviction':0.0142, 
      'layoff':0.0094, 
      'jobTraining':0.0498,
  
}

for task in tasks:
  R_values = np.full((4,27), np.nan)
  #R_values = np.zeros((4, 27))
  print("working for", task)
  for idx in range(27):
    for year in range(1, 5):
      tmp = df[(df['task']==task) & (df['year'] == year) & (df['chunk_index_without_year']==idx)]
      if(len(tmp) == 0):
        continue
      elif(len(tmp) > 1):
        assert(False)
      else:
        R_values[year-1][idx] = (tmp['R'] - four_factors[task])
  
  #mn = np.min(R_values[R_values > 0])
  #R_values = np.clip(R_values, a_min=mn, a_max=100)
  #R_values /= mn
  #R_values = np.exp(R_values)
  print(np.min(R_values), np.max(R_values))
  cmap = sns.diverging_palette(10, 133, as_cmap=True)

  with sns.axes_style("dark"):
    ax = sns.heatmap(R_values, mask=R_values==np.nan, cmap=cmap, center=0.00,
                  linewidths=.5,)
    ax.set_facecolor("white")
    #sns.heatmap(R_values,  cmap='crest')
    plt.ylabel("year")
    plt.xlabel("feature sets")
    plt.title(task)
    #plt.show()
    plt.savefig('figures/'+task+'.png')
    plt.clf()