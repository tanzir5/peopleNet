import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

binarize = True
def get_filled_columns_idx(target_columns_idx, target_row):
  filled_mask = target_row.notna()
  filled_columns_idx = []
  for idx in target_columns_idx:
    if filled_mask[idx] == False:
      continue
    c = all_columns[idx]
    if (
      (binarize==False and isinstance(target_row[c], str) and target_row[c] != 'Missing') or 
      (isinstance(target_row[c], str) == False and target_row[c] >= 0)
      ):
      filled_columns_idx.append(idx)
  return filled_columns_idx

def get_subset_cols_idx(min_features, 
                    max_features, 
                    filled_constructed_idx, 
                    filled_raw_idx):
  subset_size = np.random.randint(min_features, max_features+1)
  #print(len(filled_constructed_idx), len(filled_raw_idx))
  constructed_size = min(int(np.ceil(subset_size/3)), len(filled_constructed_idx))
  raw_size = min(subset_size-constructed_size, len(filled_raw_idx)) 
  #print("Sizes")
  #print(subset_size, constructed_size, raw_size)
  
  constructed_subset = np.random.choice(filled_constructed_idx, constructed_size, replace=False)
  raw_subset = np.random.choice(filled_raw_idx, raw_size, replace=False)
  subset_cols_idx = np.concatenate([constructed_subset, raw_subset])
  subset_cols_idx = np.sort(subset_cols_idx)
  return subset_cols_idx

def create_rows(row_num, min_features=80, max_features=150, number_of_samples=20):
  '''

  One third of the features will always come from constructed features which start with 'cm' or 'cf'.
  Constructed variables never are empty. 
  The other features will come from other non-empty feature values for this row. 

  '''

  person_texts = []
  person = df.iloc[row_num]
  filled_constructed_idx = get_filled_columns_idx(constructed_columns_idx, person)
  filled_raw_idx = get_filled_columns_idx(raw_columns_idx, person)
  for _ in range(number_of_samples):
    subset_cols_idx = get_subset_cols_idx(
      min_features, max_features, filled_constructed_idx, filled_raw_idx)
    text = ""
    for idx in subset_cols_idx:
      text += all_columns[idx]
      text += " "
      text += str(int(person[all_columns[idx]]))
      text += " "
      #text += ", "
    text += "\n"
    person_texts.append(text)

  return person_texts

def get_year(x):
  found_digit = re.search(r"\d", x)
  if found_digit is None:
    return '999'
  else:
    return x[found_digit.start()]

def create_cons_and_raw_cols_idx():
  constructed_columns_idx = []
  raw_columns_idx = []
  for idx, c in enumerate(all_columns[1:]):
    if c.startswith("cm") or c.startswith("cf"):
      constructed_columns_idx.append(idx)
    else:
      raw_columns_idx.append(idx)
  return constructed_columns_idx, raw_columns_idx

'''
data_f = open("background_split_by_lines_2.txt", "a+")
meta_data_f = open("meta_2.txt", "a+")
df = pd.read_csv('../FFChallenge_v2/background.csv')
'''

data_f = open("binary/background_split_by_lines.txt", "w+")
meta_data_f = open("binary/meta_2.txt", "w+")
df = pd.read_csv('../FFChallenge_v2/binary_background.csv')

all_columns = df.columns.tolist()
all_columns.sort(key=get_year)
constructed_columns_idx, raw_columns_idx = create_cons_and_raw_cols_idx()

for r in tqdm(range(len(df))):
  texts = create_rows(r, 60, 90, 2000)
  for t in texts:
    data_f.write(t)
  meta_data_f.write("Last done: " + str(r) + "\n")
  
data_f.close()
meta_data_f.close()