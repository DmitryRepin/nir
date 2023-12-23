import pandas as pd
from os import listdir

files = listdir(r'../preprocess_text')
df_p = []
for file in files:
    df_p.append(pd.read_csv(rf'../preprocess_text/{file}'))

df = pd.concat(df_p, ignore_index=True)
print(df.shape)