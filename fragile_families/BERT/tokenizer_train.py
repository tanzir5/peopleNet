from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = ['binary/background_split_by_lines.txt']

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=10000, show_progress=True, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
    "<sep>"
])

# Save files to disk
tokenizer.save_model("new_tokenizer")
#tokenizer.save("sv_fragile_families_tokenizer")