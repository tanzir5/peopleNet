from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertTokenizerFast

paths = ['binary/background_split_by_lines.txt']

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False)
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=11000, special_tokens=special_tokens)
tokenizer.train(paths, trainer)
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
print(cls_token_id, sep_token_id)
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

tokenizer.decoder = decoders.WordPiece(prefix="##")
new_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
new_tokenizer.save_pretrained('binary/tokenizer')

encoding = tokenizer.encode("This is one sentence.", "With this one we have a pair.")
print(encoding.tokens)
print(encoding.type_ids)

