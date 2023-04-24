from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoConfig
import numpy as np
import pandas as pd
import evaluate



def tokenize_function(examples):
    return tokenizer(examples["pair_texts"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def drop_outliers(df):
    df = df[df['label'] < 100]
    df = df[df['label'] > 0]
    return df

def create_pairs_dataset(path):
    df = pd.read_csv(path)
    df = drop_outliers(df)
    df = df.sample(frac=1)
    masked_text = df['masked_text'].tolist()
    life_span = df['label'].tolist()

    pair_texts = []
    pair_labels = []
    span_diffs = []
    shorter_goes_first = True
    for i in range(0, len(masked_text)-1, 2):
        if life_span[i] == life_span[i+1]:
            continue
        elif life_span[i] < life_span[i+1]:
            shorter_text, shorter_span = masked_text[i], life_span[i]
            longer_text, longer_span = masked_text[i+1], life_span[i+1]
        else:
            shorter_text, shorter_span = masked_text[i+1], life_span[i+1]
            longer_text, longer_span = masked_text[i], life_span[i]
        
        if shorter_goes_first:
            shorter_goes_first = False
            pair_texts.append(shorter_text +  "\n [SEP] " + longer_text)
            pair_labels.append(0)
            span_diffs.append(longer_span - shorter_span)
        else:
            shorter_goes_first = True
            pair_texts.append(longer_text +  "\n [SEP] " + shorter_text)
            pair_labels.append(1)
            span_diffs.append(longer_span - shorter_span)
    dataset_dict = {"pair_texts": pair_texts, "label": pair_labels, "diff": span_diffs}
    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.shuffle(seed=42)
    return dataset.map(tokenize_function, batched=True)


#print(train_dataset)
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
print(tokenizer.sep_token)
print(tokenizer.model_max_length)
# Define PAD Token = EOS Token = 50256

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

train_dataset = create_pairs_dataset("data/wiki_train.csv")
val_dataset = create_pairs_dataset("data/wiki_val.csv")
test_dataset = create_pairs_dataset("data/wiki_test.csv")

print(train_dataset)

model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path='bert-large-uncased', num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained('bert-large-uncased', config=model_config)
model.config.pad_token_id = model.config.eos_token_id


metric = evaluate.load("accuracy")
training_args = TrainingArguments(
                output_dir="bert_classifier_trainer", 
                evaluation_strategy="epoch",
                save_strategy="epoch",
                num_train_epochs=7,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                load_best_model_at_end=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model("bert_classifier_trainer/best")
