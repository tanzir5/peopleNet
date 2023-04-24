import pandas as pd
import numpy as np

df = pd.read_csv('data/wiki_train.csv')

text = df['masked_text'].tolist()
life_span = df['label'].tolist()
print(np.max(life_span), np.median(life_span), np.min(life_span), np.mean(life_span), np.std(life_span))

prompt = ""
count = 0
for i in range(len(text)):
  if life_span[i] > 200:
    print(life_span[i])
    print(text[i])
    count += 1
  continue
  if(len(text[i].split()) + len(prompt.split()) > 3000):
    break
  prompt += text[i]
  prompt += "\nlife span: "
  prompt += str(int(life_span[i]))
  prompt += "\n\n"
  
print(prompt)
print(i)
print(count)