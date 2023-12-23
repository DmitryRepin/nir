import re
import pandas as pd
import numpy as np

def preprocessing(row):
    url_pattern = "((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"
    if len(row) > 40 and re.search(url_pattern, row) == None:
        row = re.sub(r'#\w+', ' ', row)             # Удаление хештегов
        row = re.sub(url_pattern, ' ', row)         # Удаление ссылок
        row = re.sub(r'[^\w\s]', ' ', row)          # Удаление всего, кроме слов и пробелов
        row = re.sub(r"\b([a-zA-Z]+)\b", ' ', row)  #Удаление латиницы
        row = re.sub(r'\n+', ' ', row)              # Удаление абзацных отступов
        row = re.sub(r'\d', ' ', row)               # Удаление цифр
        row = re.sub(r'_+', ' ', row)               # Удаление нижних подчеркиваний
        row = re.sub(r' +', ' ', row)         
        row = re.sub(r' +', ' ', row)         
        row = row.strip().lower()                     
    else:
        row = np.NaN
    return row