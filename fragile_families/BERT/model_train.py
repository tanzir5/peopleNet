from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from tokenizers.implementations import ByteLevelBPETokenizer
from datasets import load_dataset

config = RobertaConfig(
    vocab_size=10000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    hidden_size=60,
)
max_seq_length=512
tokenizer = RobertaTokenizerFast(tokenizer_file="byte-level-BPE.tokenizer.json", max_len=max_seq_length)
print("tokenizer loaded")
model = RobertaForMaskedLM(config=config)
print("model parameters:", model.num_parameters())

dataset = load_dataset('text', data_files={'train': ['background_split_by_lines_2.txt'], })['train']
#dataset = load_dataset('text', data_files={'train': ['small.txt'], })['train']
#dataset = load_dataset('text', data_files={'train': ['medium.txt'], })['train']

print(dataset)

def tokenize_function(examples):
    # Remove empty lines
    examples['text'] = [
        line for line in examples['text'] if len(line) > 0 and not line.isspace()
    ]
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=max_seq_length,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
    )

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    desc="Running tokenizer on dataset line_by_line",
)
tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.05)
print(tokenized_datasets)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

print("collator loaded")

training_args = TrainingArguments(
    output_dir="./Models",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=64,
    save_strategy="steps",
    save_steps=10000,
    eval_steps=10000,
    save_total_limit=10,
    prediction_loss_only=True,
    per_device_eval_batch_size=64,
    evaluation_strategy="steps",
    logging_steps=100,
    load_best_model_at_end=True,

)

print("arguments loaded")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

print("trainer loaded")

trainer.train()

trainer.save_model("./final_models")