from pymystem3 import Mystem
import pandas as pd
from os import listdir
from preprocessing import preprocessing

m = Mystem()

def lemmatize(list):
    batch_size = 500
    text_batch = [list[i: i + batch_size] for i in range(0, len(list), batch_size)]
    res = []
    for text in text_batch:
        merged_text = '|'.join(text)
        doc = []

        words = m.lemmatize(merged_text)
        for t in words:
            if t != '|':
                doc.append(t)
            else:
                res.append(preprocessing(''.join(doc)))
                doc = []
        
        res.append(preprocessing(''.join(doc)))
        print(f'Обработано {len(res)} строк')
    
    return res

if __name__ == "__main__":
    files = listdir(r'../preprocess_text')

    for file in files:
        print(f'Обработка файла {file}')
        df = pd.read_csv(rf'..\preprocess_text\{file}')
        list_text = df['text'].tolist()
        print(f'Количество строк {len(list_text)}')
        result = lemmatize(list_text)
        df['lemma_text'] = result
        print(df.head(3))
        df.to_csv(fr'../preprocess_text/{file}', index=False, header=True)