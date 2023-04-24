import pandas as pd

file_path = '../Data/arxiv-metadata-oai-snapshot.json'
json_file = open(file_path, "r")
 
#read whole file to a string
#data = json_file.read()
#print("YO")
df = pd.read_json(file_path, lines=True, nrows=100000)
authors = df['authors_parsed']
freq = {}
for cur in authors:
  for author in cur:
    author = ' '.join(author)
    if author not in freq:
      freq[author] = 0
    freq[author] += 1

val = list(freq.values())
f = {}
for v in val:
  if v not in f:
    f[v] = 0
  f[v] += 1
print(val)  