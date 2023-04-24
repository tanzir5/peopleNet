from datasets import load_dataset
from datasets import Dataset
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertConfig
from transformers import BertTokenizerFast
from transformers import BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from tqdm import tqdm

import ast
import os
import pandas as pd 

background_df = pd.read_csv('../../Analysis/binary_background.csv')
CHALLENGE_ID = 'challengeID'
VOCAB_SIZE = 300

def create_text_df(chunk):
  #print(chunk)
  #print(background_df)
  chunk.append(CHALLENGE_ID)
  new_chunk = []
  bad_count = 0
  for col in chunk:
    if col in background_df.columns:
      new_chunk.append(col)
    else:
      bad_count += 1
  if bad_count > 10:
    return None
  chunk = new_chunk
  small_df = background_df[chunk]
  texts = []
  ids = []
  for i in range(len(small_df)):
    person = small_df.iloc[i]
    text = ""
    for col in chunk:
      if col == CHALLENGE_ID:
        continue
      val = person[col]
      if pd.notna(val):
        val = int(val)
        val = col + '_' + str(val)
      else:
        val = col +'_#'
      text += val
      text += " "
    texts.append(text)
    ids.append(person[CHALLENGE_ID])
  return pd.DataFrame({CHALLENGE_ID: ids, 'text':texts})

def create_dataset(texts):
  text_dict = {'text': texts}
  dataset = Dataset.from_dict(text_dict)
  print(dataset)
  return dataset

def create_tokenizer(texts):
  tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
  tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False)
  tokenizer.pre_tokenizer = pre_tokenizers.CharDelimiterSplit(' ')
  special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
  trainer = trainers.WordLevelTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
  tokenizer.train_from_iterator(texts, trainer=trainer)
  tokenizer.post_processor = processors.TemplateProcessing(
      single="[CLS] $A [SEP]",
      pair="[CLS] $A [SEP] $B:1 [SEP]:1",
      special_tokens=[
          ("[CLS]", tokenizer.token_to_id("[CLS]")),
          ("[SEP]", tokenizer.token_to_id("[SEP]")),
      ],
  )
  #tokenizer.decoder = decoders.WordPiece(prefix="##")
  new_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
  return new_tokenizer

def save_eth(target_dir, tokenizer, text_df):
  tokenizer.save_pretrained(target_dir+'tokenizer')
  text_df.to_csv(target_dir+'text.csv')

global_tokenizer = None
def tokenize_function(examples):
  # Remove empty lines
  examples['text'] = [
      line for line in examples['text'] if len(line) > 0 and not line.isspace()
  ]
  return global_tokenizer(
      examples['text'],
      padding='max_length',
      truncation=True,
      max_length=max_seq_length,
      # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
      # receives the `special_tokens_mask`.
      return_special_tokens_mask=True,
  )

def create_args(target_dir):
  training_args = TrainingArguments(
    output_dir=target_dir+"/Models",
    overwrite_output_dir=True,
    num_train_epochs=40,
    per_device_train_batch_size=64,
    save_strategy="steps",
    save_steps=200,
    eval_steps=200,
    save_total_limit=1,
    prediction_loss_only=True,
    per_device_eval_batch_size=64,
    evaluation_strategy="steps",
    logging_steps=200,
    load_best_model_at_end=True,
  )
  return training_args 

max_seq_length=80
config = BertConfig(
  vocab_size=VOCAB_SIZE,
  max_position_embeddings=max_seq_length,
  num_attention_heads=3,
  num_hidden_layers=2,
  hidden_size=12,
)
chunk_df = pd.read_csv('../../Analysis/chunks.csv')
for i, chunk in tqdm(enumerate(chunk_df['chunk'].tolist())):
  chunk_idx = chunk_df['chunk_index'].iloc[i]
  target_dir = str(chunk_idx) + "/"
  if os.path.exists(target_dir) == False:
    os.mkdir(target_dir)
  else:
    continue
  
  chunk = ast.literal_eval(chunk)
  text_df = create_text_df(chunk)
  if text_df is None:
    continue
  tokenizer = create_tokenizer(text_df['text'].tolist())
  dataset = create_dataset(text_df['text'].tolist())
  model = BertForMaskedLM(config=config)
  global_tokenizer = tokenizer
  tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    desc="Running tokenizer on dataset line_by_line",
  )
  tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.01)
  data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
  )

  training_args = create_args(target_dir)
  trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
  )
  trainer.train()
  trainer.save_model(target_dir+"final_models")
  save_eth(target_dir, tokenizer, text_df)

