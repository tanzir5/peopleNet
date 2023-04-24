import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

#test_small = True
test_small = False
if test_small:
  LIM = 100
else:
  LIM = 1000000000

def get_year(x):
  found_digit = re.search(r"\d", x)
  if found_digit is None:
    return '999'
  else:
    return x[found_digit.start()]


def get_valid_entries_count(col):
  #print(df[col])
  tmp = df[col][df[col].notna()]
  count = 0
  values = []
  for entry in tmp:
    if (
        (isinstance(entry, str) and entry != 'Missing') or 
        (isinstance(entry, str) == False and entry >= 0)
        ):
        count += 1
        values.append(entry)
  
  unique_count = 0
  std = "NA"
  if len(values) > 1:
    values = np.array(values)
    unique_count = len(set(values))
    if all(np.array([isinstance(i, np.number) for i in values])):
      std = np.std(values)
    else:
      std = "NA"
  return count, unique_count, std

def create_same_cols(df):
  all_columns = df.columns.tolist()
  all_columns.sort(key=get_year)
  freq = {}
  entry_count = {}
  features = []
  years = []
  totals = []
  uniques = []
  stds = []
  same_cols = {}
  for i, col in tqdm(enumerate(all_columns)):
    col_year = get_year(all_columns[i])
    all_columns[i] = (all_columns[i], col_year)
    if col_year not in freq:
      freq[col_year] = 0
    freq[col_year] += 1
    total, unique, std = get_valid_entries_count(col)
    features.append(col)
    years.append(col_year)
    totals.append(total)
    uniques.append(unique)
    stds.append(std)
    if unique > 0 and '1' <= col_year <= '5':
      col_without_year = col.replace(col_year, '', 1)
      if col_without_year not in same_cols:
        same_cols[col_without_year] = []
      same_cols[col_without_year].append(col)
    #print(col, entry_count[col])
    #break
  #new_df = pd.DataFrame({'feature':features, 'year':years, 'total':totals, 'unique':uniques, 'stds':stds})
  #new_df.to_csv('valid_entries.csv', index=False)
  #exit(0)
  return same_cols

def create_cols_by_year_combination(df, same_cols):
  good_sets = 0
  bad_sets = 0
  cols_by_year_combination = {}
  for key in same_cols:
    if len(same_cols[key]) > 1:
      good_sets += 1
      current_years = set()
      for col in same_cols[key]:
        current_years.add(get_year(col))
      current_years = sorted(current_years)
      current_years = frozenset(current_years)
      if current_years not in cols_by_year_combination:
        cols_by_year_combination[current_years] = []
      cols_by_year_combination[current_years].append(key) 
    else:
      bad_sets += 1
  return cols_by_year_combination

def create_chunks(df, cols_by_year_combination, same_cols):
  chunks = []
  chunk_years = []
  chunk_combs = []
  chunk_indices = []
  chunk_indices_without_year = []
  chunk_count = 0
  chunk_index_without_year = 0
  for year_comb, cols in cols_by_year_combination.items():
    year_comb_int = int(''.join(sorted(list(year_comb))))
    for year in year_comb:
      for i in range(0,len(cols)-30,60):
        current_cols = cols[i:i+60]
        current_chunk = []
        for col_name in current_cols:
          for col in same_cols[col_name]:
            if get_year(col) == year:
              current_chunk.append(col)
        chunks.append(current_chunk)
        chunk_years.append(year)
        chunk_combs.append(year_comb_int)
        chunk_indices.append(chunk_count)
        chunk_indices_without_year.append(chunk_index_without_year)
        chunk_count += 1
    chunk_index_without_year += 1
  
  chunk_df = pd.DataFrame({
    'year':chunk_years, 
    'comb':chunk_combs, 
    'chunk_index':chunk_indices,
    'chunk_index_without_year':chunk_indices_without_year,
    'chunk':chunks, 
    })
  #print(chunk_count)
  chunk_df.to_csv('chunks.csv')

df = pd.read_csv('../FFChallenge_v2/background.csv', nrows=LIM)
same_cols = create_same_cols(df)
cols_by_year_combination = create_cols_by_year_combination(df, same_cols)
create_chunks(df, cols_by_year_combination, same_cols)
