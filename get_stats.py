import pandas as pd
from os import listdir

if __name__ == "__main__":
    files = listdir('preprocess_text')
    nmb_lines = 0
    nmb_words = 0

    for file in files:
        nmb_words_class = 0

        df = pd.read_csv(f'preprocess_text/{file}')
        nmb_lines += df.shape[0]
        for index, row in df.iterrows():
            nmb_words_class += len(row['no_stop_words'].split())
        nmb_words += nmb_words_class
        print(f'Статистика по файлу {file}')
        print(f'Количество текстов {df.shape[0]}')
        print(f'Количестов слов {nmb_words_class}')
        print(40 * '-')

    print('Итого:')
    print(f'Количество текстов {nmb_lines}')
    print(f'Количество слов {nmb_words}')