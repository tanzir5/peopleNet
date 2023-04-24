from transformers import AutoModelForSequenceClassification
from transformers import RobertaTokenizerFast
import torch
from torch import nn
alt_CLS = "#"

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = torch.mean(features,1)  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

def tokenize(texts, tokenizer):
  print(texts)
  #texts = alt_CLS + " " + texts
  print(texts)
  #for i in range(len(texts)):
  #  texts[i] = "<s> " + texts[i]
  return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=True,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
    )

num_labels = 1
max_seq_length = 512
tokenizer = RobertaTokenizerFast(tokenizer_file="../byte-level-BPE.tokenizer.json", max_len=max_seq_length)
model = AutoModelForSequenceClassification.from_pretrained('../final_models', num_labels=num_labels)
model.classifier = ClassificationHead(model.config)
for name, param in model.named_parameters():
    print(name)
    #if 'classifier' not in name: # classifier layer
    #  param.requires_grad = False
#return model 
print("*"*100)
print(model.classifier)
#print(model)
#print(model)
exit(0)
text = "f1a7 0 , m1a7 1 , m1a8 2 , m1a10 1 , m1a16 1 , cm1age 27 , m1b0 2 , m1b22c 1 , m1b22d 1 , m1b23d 3 , m1b26 1 , m1b28 2 , m1c1a 2 , m1c2 2 , m1c4a 1 , m1c5a 1 , m1c7 1 , m1d2c 1 , m1e1e1 2 , m1e3a 2 , m1e4a 1 , m1e4b 1 , cm1adult 2 , m1f11a 2 , m1g2 5 , m1g5 2 , m1i2a 38 , m1i2b 1682.41560210582 , m1i3 6 , m1i8 2 , m1i9 2 , m1j2a 0.0382624278170366 , m1j4 2 , cm1finjail 0.0 , cmf1finjail 0.0 , cm1span 0 , cf1fint 0 , cf1edu 3 , m1b25 9.39506975438657 , cm2mint 0 , cm2samp 7 , cf2fint 0 , cc2natsmx 0.0 , cm3fdiff 0 , cm3samp 5 , cf3fint 0 , cf3samp 5 , cc3citsm 0.0 , cm4samp 5 , m4c7 2 , m4c24 2 , cf4samp 7 , cc4natsm 0.0 , cc4natsmx 0.0 , cm5mint 0 ,"
#print(tokenizer.special_tokens)
tokenized_text = torch.tensor(tokenize(text, tokenizer)['input_ids']).unsqueeze(0)
print(tokenized_text.size(), tokenized_text[0][:10])
exit(0)
#print(type(tokenized_text['input_ids']))
out = model(tokenized_text)
print("SDF", out[0].size())
print(type(out), len(out), out['last_hidden_state'].size())
#print(tokenized_text)
#print(model)