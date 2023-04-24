from datasets import load_dataset
dataset = load_dataset("wiki_bio")
dataset = dataset['train']
print(dataset)
#dataset = dataset.shard(num_shards=100, index=98)
#print(dataset)
a = dataset['input_text']
print("ok")
count = 0
birth_words = ['born', 'date of birth']
death_words = ['died', 'date of death']
for x in a:
  for i, y in enumerate(x['table']['content']):
    splitted_words = y.split()
    if 'aged' in splitted_words:
      idx = splitted_words.index('aged')
      if idx != len(splitted_words)-1 and splitted_words[idx+1].isnumeric():
        if (x['table']['column_header'][i] == 'death_date'):
          count += 1
          break
      else:
        pass
        #print("WRONG")
        #print(y)
print(count, count/len(a))