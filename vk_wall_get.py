import requests
import pandas as pd

def wall_get():
    TOKEN_USER = 'vk1.a.G_y24jyCE33knAATh7q5Gub3Dom5f9T2IhIXmDZfTzoV04YArnLDyo7zb2aUX1Wcy8nyKfemaE_Ufus2omqDRdURvsXgejcsUglII-SfMulROrlkOP6jOfZIePkuX3ReIr77wC6CcsC6rSM1z61K-kigsYtrlhFaRHoUTOIbf-Hx6XJKhwL6IxP78mFZMozZ6bdU1nZQuxxqKrxmp8qBWA'
    VERSION = '5.199'
    DOMAIN = 'live_fast_die_youngg'

    # через api vk вызываем статистику постов
    response = requests.get('https://api.vk.com/method/wall.get',
    params={'access_token': TOKEN_USER,
            'v': VERSION,
            'domain': DOMAIN,
            'count': 1,
            'filter': str('owner')})
    
    count = response.json()['response']['count']
    data = []
    print(f' [x] Записей в сообществе - {count}\n')

    for i in range(count // 100 + 1):
        response = requests.get('https://api.vk.com/method/wall.get',
        params={'access_token': TOKEN_USER,
            'v': VERSION,
            'domain': DOMAIN,
            'offset': i * 100,
            'count': 100,
            'filter': str('owner')})
        
        data_100 = response.json()['response']['items']
        data = data + data_100
        print(f' [x] Загружено записей: {len(data)}')

    text = []

    for post in data:
        text.append(post['text'])           

    df = pd.DataFrame({
        'text': text
    })

    df.to_csv(rf'clean_text/{DOMAIN}', columns=["text"], index=False)


if __name__ == "__main__":
    wall_get()