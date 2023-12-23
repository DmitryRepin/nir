from nltk.corpus import stopwords
from os import listdir
import pandas as pd

rus_stopwords = stopwords.words("russian")

def remove_stopwords(list_text):
    new_list_text = []
    for row in list_text:
        new_words = []
        words = row.split(' ')
        for word in words:
            if word not in rus_stopwords:
                new_words.append(word)
            
        new_row = ' '.join(new_words)
        new_list_text.append(new_row)
    
    return new_list_text


if __name__ == "__main__":
    files = listdir(r'../preprocess_text')

    for file in files:
        print(f'Обработка файла {file}')
        df = pd.read_csv(rf'../preprocess_text/{file}')
        df = df.dropna()
        list_text = df['lemma_text'].tolist()
        df['no_stop_words'] = remove_stopwords(list_text)
        df.to_csv(fr'../preprocess_text/{file}', index=False, header=True)