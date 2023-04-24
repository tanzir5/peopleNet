import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

#data_f = open("background_split_by_lines_2.txt")
data_f = open("binary/background_split_by_lines.txt")
lines = data_f.readlines()
texts = []
family_ids = []
for i,line in tqdm(enumerate(lines)):
  current_family = int((i/2000))
  texts.append(line)
  family_ids.append(current_family)

df = pd.DataFrame({'family_id': family_ids, 'text': texts})
df.to_csv('binary/background_x2000.csv', index=False)
