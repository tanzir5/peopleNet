from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import GPT2Config
import numpy as np
import evaluate

def tokenize_function(examples):
    return tokenizer(examples["masked_text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)

#print(train_dataset)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(tokenizer.model_max_length)
tokenizer.padding_side = "left"
# Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token

train_dataset = load_dataset("csv", data_files="data/wiki_train.csv")['train'].map(tokenize_function, batched=True)
val_dataset = load_dataset("csv", data_files="data/wiki_val.csv")['train'].map(tokenize_function, batched=True)
test_dataset = load_dataset("csv", data_files="data/wiki_test.csv")['train'].map(tokenize_function, batched=True)


model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='gpt2', num_labels=1)
model = AutoModelForSequenceClassification.from_pretrained('gpt2', config=model_config)
model.config.pad_token_id = model.config.eos_token_id

metric = evaluate.load("mae")
training_args = TrainingArguments(
                output_dir="test_trainer", 
                evaluation_strategy="epoch",
                save_strategy="epoch",
                num_train_epochs=7,
                #per_device_train_batch_size=8,
                #per_device_eval_batch_size=8,
                save_total_limit=5,
                load_best_model_at_end=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model("test_trainer/best")
