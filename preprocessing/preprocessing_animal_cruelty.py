import pandas as pd
from preprocessing import preprocessing

if __name__ == "__main__":
    df = pd.read_csv(r'../clean_text/zhivoderam_net')
    print(df.shape)
    df = df.dropna()
    df = df.drop_duplicates()
    print(df.shape)
    df['text'] = df['text'].map(preprocessing)
    df = df.dropna()
    print(df.shape)
    df = df.loc[df['text'] != '']
    df['label'] = '__label__animal_cruelty'
    print(df.shape)
    df.to_csv(r'../preprocess_text/animal_cruelty', index=False, header=True)