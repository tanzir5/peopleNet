import pandas as pd
goods = [13, 14, 16]

CHALLENGE_ID = 'challengeID'

train_df = pd.read_csv('../train.csv')
train_df = train_df[(
  (train_df[CHALLENGE_ID] == 13) | 
  (train_df[CHALLENGE_ID] == 14) | 
  (train_df[CHALLENGE_ID] == 16)
  )]
train_df.to_csv('../small_train.csv')

print("DONE TRAIN")

background_df = pd.read_csv('../background_x2000.csv')
background_df = background_df[(
  (background_df['family_id'] == 12) | 
  (background_df['family_id'] == 13) | 
  (background_df['family_id'] == 15)
  )]
print(len(background_df))
background_df.to_csv('../small_background_x2000.csv')

