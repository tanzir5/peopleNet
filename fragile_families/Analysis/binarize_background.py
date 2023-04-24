import pandas as pd
import numpy as np
from tqdm import tqdm
#TEST_SMALL = True
TEST_SMALL = False
if TEST_SMALL:
  LIM = 100
else:
  LIM = 1000000000

CHALLENGE_ID = 'challengeID'
df_dict = {}
def add_col(values, below_smaller, median):
  binary_values = []
  for val in values:
    if pd.isna(val):
      val = np.nan
    else:
      if (below_smaller and val <= median) or (below_smaller==False and val < median):
        val = 0
      else:
        val = 1
    binary_values.append(val)
  df_dict[col] = binary_values
  
df = pd.read_csv('../FFChallenge_v2/background.csv', nrows=LIM)

types = set()
bad_count = 0 
good_count = 0
good_cols = []
for col in tqdm(df.columns):
  if col == CHALLENGE_ID:
    df_dict[col] = df[col]
    continue

  if str(df[col].dtype) == 'object':
    bad_count += 1
  else:
    values = df[col].to_numpy()
    values = values.astype('float')
    values[values < 0.0] = np.nan
    nan_percent = np.sum(np.isnan(values)/len(values)) * 100 
    if nan_percent > 99:
      continue
    median = np.nanmedian(values)
    #continue
    below = np.sum(np.array(values < median))
    med = np.sum(np.array(values == median))
    above = np.sum(np.array(values > median))
    below_smaller = below < above
    if below < above:
      below += med
    else:
      above += med
    if below >= 40 and above >= 40:
      good_count += 1
      add_col(values, below_smaller, median)

print(good_count, "y")
binary_background_df = pd.DataFrame.from_dict(df_dict)
binary_background_df.to_csv('binary_background.csv', index=False)