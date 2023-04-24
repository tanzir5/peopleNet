from transformers import pipeline
from transformers import RobertaTokenizerFast
import numpy as np
import torch
from tokenizers.processors import BertProcessing
from tokenizers.implementations import ByteLevelBPETokenizer
from tqdm import tqdm
import pandas as pd

def get_mean_dist_from_centroid(embs):
  # embs should be sample_size x dimension i.e. 10 x 60 right now
  centroid = np.mean(embs, axis=0)
  total_dist = 0
  for emb in embs:
    total_dist += np.linalg.norm(centroid-emb)
  return total_dist/len(embs)

def custom_strlist_to_npfloats(a_str):
  a_str = a_str[1:-1]
  a_float = [float(x) for x in a_str.split()]
  a_np = np.array(a_float)  
  return a_np  

sample_size = 100
repeat_per_person = 250
max_lim = 1e18
df = pd.read_csv('background_embs_x2000.csv')#, nrows=max_lim)

person_rows = {}
counter = 0
num_persons = 0
for i in tqdm(range(4242)):
  if counter + 2000 > max_lim:
    num_persons = i
    break
  person_rows[i] = []
  for k in range(2000):
    person_rows[i].append(counter)
    #assert(int(df.iloc[counter]['family_id']) == i)
    counter += 1

if(num_persons == 0):
  num_persons=4242

real_embs = []
fake_embs = []


fake_dists = []
real_dists = []

for i in tqdm(range(num_persons)):
  for _ in range(repeat_per_person):
    picked_samples = np.random.choice(person_rows[i], sample_size, replace=False)
    current_embs = []
    for idx in picked_samples:
      emb = custom_strlist_to_npfloats(df.iloc[idx]['emb'])
      current_embs.append(emb)
    current_embs = np.array(current_embs)
    real_embs.append(current_embs)
    real_dists.append(get_mean_dist_from_centroid(current_embs))

for _ in tqdm(range(len(real_embs))):
  picked_people = np.random.choice(range(num_persons), sample_size, replace=False)
  picked_rows = np.random.choice(range(2000), sample_size)
  current_embs = []
  for i in range(sample_size):
    person_id = picked_people[i]
    row_num = picked_rows[i]
    final_row_id = person_rows[person_id][row_num]
    emb = custom_strlist_to_npfloats(df.iloc[final_row_id]['emb'])
    current_embs.append(emb)
  current_embs = np.array(current_embs)
  fake_embs.append(current_embs)
  fake_dists.append(get_mean_dist_from_centroid(current_embs))

real_embs = np.array(real_embs)
fake_embs = np.array(fake_embs)
np.save('real_dists_250_samples_100.npy', real_dists)
np.save('fake_dists_250_samples_100.npy', fake_dists)
np.save('real_embs_250_samples_100.npy', real_embs)
np.save('fake_embs_250_samples_100.npy', fake_embs)