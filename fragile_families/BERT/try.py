from transformers import pipeline
from transformers import RobertaTokenizerFast
import numpy as np
import torch
from tokenizers.processors import BertProcessing
from tokenizers.implementations import ByteLevelBPETokenizer

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

def stats(a):
    return a.shape, np.min(a), np.median(a), np.max(a), np.mean(a), np.std(a)

repeat_per_person = 100
num_persons = 3

'''
real_2d = np.load('real_small_2d.npy')
print(real_2d.shape)
#real_2d = real_2d[10000:10050]
np.random.shuffle(real_2d)
print(real_2d.shape)
x = [row[0] for row in real_2d]
y = [row[1] for row in real_2d]
person_id = [str(int(i/repeat_per_person)) for i in range(len(real_2d))]
df = pd.DataFrame({'x':x, 'y':y, 'person':person_id})
print(real_2d.shape)
print(df.columns)
sns.scatterplot(data=df, x='x', y='y', hue='person')
plt.show()
exit(0)


real_embs = np.load('real_embs_samples_100.npy')
print(real_embs.shape)

real_embs_2 = np.reshape(real_embs, (real_embs.shape[0]*real_embs.shape[1], -1))
print(real_embs_2.shape)


st = 200000
ed = st + repeat_per_person * num_persons
real_embs_2 = real_embs_2[st:ed]
print("real embs dim:", real_embs.shape)
print(real_embs_2.shape)
#print(real_embs[3][8] == real_embs_2[38])
real_2d = TSNE(n_components=2, learning_rate='auto',
                   init='random', perplexity=3, verbose=1, n_iter=10000).fit_transform(real_embs_2)
np.save('real_small_2d.npy', real_2d)
exit(0)
'''
real = np.load('real_dists_10_samples_100.npy')
fake = np.load('fake_dists_10_samples_100.npy')

#print(real)
#print(fake)

print(stats(real))
print(stats(fake))


sns.histplot(real, color='b', kde=True)
sns.histplot(fake, color='r', kde=True)
plt.xlabel("Mean Distance from Cluster Centroid")
plt.legend(labels=['Cluster created from\n the same person', 'Cluster created from\n different persons'])
plt.show()