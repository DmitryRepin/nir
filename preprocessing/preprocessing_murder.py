import pandas as pd
from preprocessing import preprocessing

if __name__ == "__main__":
    df_0 = pd.read_csv(r'../clean_text/maniac111')
    print(df_0.shape)
    df_0 = df_0.dropna()
    df_0 = df_0.drop_duplicates()
    print(df_0.shape)
    df_0['text'] = df_0['text'].map(preprocessing)
    df_0 = df_0.dropna()
    print(df_0.shape)
    df_0 = df_0.loc[df_0['text'] != '']
    df_0['label'] = '__label__murder'
    print(df_0.shape)
    df_1 = pd.read_csv(r'../clean_text/skillers_world')
    print(df_1.shape)
    df_1 = df_1.dropna()
    df_1 = df_1.drop_duplicates()
    print(df_1.shape)
    df_1['text'] = df_1['text'].map(preprocessing)
    df_1 = df_1.dropna()
    print(df_1.shape)
    df_1 = df_1.loc[df_1['text'] != '']
    df_1['label'] = '__label__murder'
    print(df_1.shape)
    df = pd.concat([df_0, df_1], ignore_index=True)
    df.to_csv(r'../preprocess_text/murder', index=False, header=True)