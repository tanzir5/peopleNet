from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tqdm import tqdm
import numpy as np
def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

tokenizer = ByteLevelBPETokenizer(
    "tokenizer/sm_fragile_families_tokenizer-vocab.json",
    "tokenizer/sm_fragile_families_tokenizer-merges.txt",
)

#tokenizer = Tokenizer.from_file("")
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

f = open("background_split_by_lines_2.txt")
done = []
with open('background_split_by_lines_2.txt') as f:
    for num, line in tqdm(enumerate(f)):
        if np.random.randint(100) != 0:
            continue
        splitted_line = line.split()
        feature_numbers = len(splitted_line)/3
        encoded = tokenizer.encode(line)
        decoded = tokenizer.decode(encoded.ids).split()
        for i in range(len(decoded)-2, 0, -1):
            if isfloat(decoded[i]) == False and decoded[i] != ",":
                if decoded[i] not in splitted_line:
                    assert(False)
                idx = splitted_line.index(decoded[i])/3
                if idx < feature_numbers - 1:
                    print(idx, feature_numbers)
                    input()
                break

done = np.array(done)
exit(0)

output = tokenizer.encode(line)
print(type(output))
#print(output.ids)
print(len(output.ids))
print(len(line.split()))
print(output)
print(output.tokens)
print("x"*100)
print(tokenizer.decode(output.ids))
# Encoding(num_tokens=7, ...)
# tokens: ['<s>', 'Mi', 'Ġestas', 'ĠJuli', 'en', '.', '</s>']