from os import listdir
import pandas as pd
import re
from sklearn.model_selection import train_test_split

def preprocessing():
    files = listdir(r'../preprocess_text')
    df_p = []
    for file in files:
        df_p.append(pd.read_csv(rf'../preprocess_text/{file}'))

    df = pd.concat(df_p, ignore_index=True)
    df['label_text'] = df['label'] + ' ' + df['no_stop_words']
    train, test = train_test_split(df, test_size = 0.2)

    nmb_words_class = 0
    for index, row in train.iterrows():
        nmb_words_class += len(row['label_text'].split()) - 1
    print('Статистика по файлу trigger_warnings.train')
    print(f'Количество текстов {train.shape[0]}')
    print(f'Количестов слов {nmb_words_class}')
    print(40 * '-')

    nmb_words_class = 0
    for index, row in test.iterrows():
        nmb_words_class += len(row['label_text'].split()) - 1
    print('Статистика по файлу trigger_warnings.train')
    print(f'Количество текстов {test.shape[0]}')
    print(f'Количестов слов {nmb_words_class}')
    print(40 * '-')

    train.to_csv(r'..\for_train\trigger_warnings.train', columns=["label_text"], index=False, header=None)
    test.to_csv(r'..\for_train\trigger_warnings.test', columns=["label_text"], index=False, header=None)
    
if __name__ == "__main__":
    preprocessing()