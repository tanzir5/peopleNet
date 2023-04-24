from transformers import BertTokenizerFast, AutoModel
from tokenizers import decoders
import torch
import numpy as np 

def tokenize(texts):
  return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_seq_length,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
    )
max_seq_length = 512
tokenizer = BertTokenizerFast.from_pretrained('../binary/tokenizer')
#print(tokenizer.decoder)
decoder = decoders.WordPiece(prefix="##")
#encoding = tokenizer(["This is one sentence.", "With this one we have a pair.", "bla bla"])
texts = ["m1f5 1 f1b26 0 f1b28 0 cf1adult 0 f1f15 0 m2a9a 0 m2a12 0 m2b31f 0 m2c37a4 0 m2k18f1c 0 cm2povco 0 cf2new12 0 cf2marm 0 f2fb33b 0 f2h7b 0 f2j10 0 f2l6a 0 cf2hhimp 0 cf2povco 0 cm3b_age 0 cm3cohf 0 m3b4f 0 m3b15_7 0 m3b17 0 m3c16p 0 m3c44 0 cf3fint 0 cf3new30 0 f3b7a 0 f3e2f1 0 cf3marp 0 cf3adult 0 f3i4b 0 f3i8a2 0 f3i17 0 f3j24 0 f3j62 0 cf3alc_case 0 f3l6a 0 f3l9c 0 cm4amrf 0 m4b3 0 m4b3d 0 m4b4b18 0 m4b25 0 m4c28 0 m4d1d 0 m4f4b 0 m4j2b1_3 0 f4a3b1b_5 0 f4e8 0 cf4kids 0 f4h1n 0 f4i7c 0 cf4hhimp 0 cf4povcob 0 cf4povcab 0 k5a3d 0 k5g2h 0 m5b3 0 m5g0 0 m5a3i_1 0 cm5gmom 0 m5e9_3 0 m5a6g01_103 0 m5a6g02_103 0 m5b22_108 0 f5a101 0 f5b2c 0 f5b12b 0 f5f8c1 0 "]
texts.append("m1b9b2 1 cm1marf 0 m1d2d 0 cm1gdad 0 m1g5 1 cm1edu 0 f1lenmin 0 f1e6d1 0 f1b25 0 cm2fint 1 m2c5 1 m2d2c 0 m2f2b4 1 cm2gmom 0 m2l10 1 f2a8b 1 f2b2 1 f2b27f 1 f2c6c 0 f2d1a 1 cf2kids 1 f2g7a2 1 f2k5a9 1 cf2hhinc 0 kind_d2f 1 cm3fint 1 m3a12c 1 cm3marf 0 m3f3b10 1 m3j43 1 f3b6d 1 f3c10b 1 f3c19 1 cf3adult 1 cf3gdad 1 f3i10 1 f3r8 1 f3k27f 1 f3l9c 1 cf3hhimpb 1 hv3a9 0 hv3m9 1 hv3m17 0 cm4mint 1 cm4relf 0 m4b4a2 1 m4c31a 1 m4f2d2 1 m4f3a2_4 1 m4h1p 1 m4j14 1 m4k3a_1a 1 m4l9c 1 cm4tele 1 m4citywt_rep10 0 cf4natsm 1 cf4natsmx 1 f4f2f10 1 f4k3a_1a 1 f4l9c 1 cf4hhinc 0 cf4hhimp 1 hv4d1b 1 hv4f1g 0 hv4k0 0 m5a10a 1 m5f29 1 m5j6i 1 m5a3i_12 1 m5a6g03_91 1 m5i3a_13 1 m5a3a1_101 1 f5g3 1 f5k5h 1 f5f16a_2 1 f5i3a_9 1 n5c3e 1 n5a9_6 1 p5h16b 1 p5l1 1 p5m4 1 p5q1m 1 p5q3da 1 ")
texts.append("m1a16 0 m1b10a2 0 cm1relf 0 m1h3b 0 cm1povca 0 cm1finjail 0 f1c2f 0 f1c7 0 cf1kids 0 cf1gdad 0 f1h4b 0 cm2relf 0 cm2cohf 0 m2d2e 0 m2e8k 0 cm2hhimp 0 cf2new12 0 cf2samp 1 f2a11 0 f2b37f 0 f2c3k 0 f2d2c 0 f2d7c 0 cf2marp 0 f2h1 0 f2k22f1d 0 f2l6a 0 cm3mint 0 m3b32k 0 m3c3g 0 m3c9 0 m3e25 0 cm3marp 0 m3i0p 0 m3j3 0 m3k3a_6 0 m3k25a1 0 cm3hhinc 0 cm3hhimp 0 f3e14 0 f3e19a 0 f3f2f6 0 f3i18a 0 cf3hhincb 0 m4e8 0 m4e10 0 m4e21a 0 m4f2e8 0 cm4adult 0 m4h2a_7 0 m4i0n1 0 m4j24b_6 0 m4k3a_12 0 m4l2 0 f4a8b_4 0 cf4age 0 f4b4b1 0 f4e18g 0 f4f2c10 0 f4i18b 0 f4j13 0 f4k24a 0 cf4hhimp 0 m5a5b08 0 m5a6e 0 m5a6g01_6 0 m5b17d_3 0 m5a3a1_101 0 f5a5d09 0 f5a8b05 0 f5b6a1 0 f5d2f2 0 f5f18b 0 f5i4 0 ")
print(len(texts[0].split()))
encoding = tokenize(texts)
print(encoding['input_ids'][0])
model_path = '../binary/final_models'
model = AutoModel.from_pretrained(model_path)
out = model(torch.tensor(encoding['input_ids']))
print(model)
x = out['last_hidden_state'][0][0] #cls
'''print(out['last_hidden_state'][0][1]) #m1f5
print(out['last_hidden_state'][0][2]) #1
print(out['last_hidden_state'][0][3]) #f1b26
print(out['last_hidden_state'][0][4]) #0
print(out['last_hidden_state'][0][5]) #f1b28
print(out['last_hidden_state'][0][6]) #0 '''
y = out['last_hidden_state'][1][0]
z = out['last_hidden_state'][2][0]

diff1 = x.detach().numpy() - y.detach().numpy() 
#print(np.min(diff), np.median(diff), np.max(diff), np.mean(diff), np.std(diff))
diff2 = x.detach().numpy() - z.detach().numpy() 
#print(np.min(diff), np.median(diff), np.max(diff), np.mean(diff), np.std(diff))
diff3 = y.detach().numpy() - z.detach().numpy() 
#print(np.min(diff), np.median(diff), np.max(diff), np.mean(diff), np.std(diff))

diff = diff1 + diff3
print(np.min(diff), np.median(diff), np.max(diff), np.mean(diff), np.std(diff))

#print(encoding[0])
#print(tokenizer.decode(encoding[2].ids))