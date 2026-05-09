import os
import urllib.request

DATA_DIR = 'data'
BASE_URL = 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv'

os.makedirs(DATA_DIR, exist_ok=True)

for split in ['train', 'test']:
    path = f'{DATA_DIR}/{split}.csv'
    if os.path.exists(path):
        print(f'{split}.csv already exists, skipping')
        continue
    url = f'{BASE_URL}/{split}.csv'
    print(f'Downloading {split}.csv...')
    urllib.request.urlretrieve(url, path)
    print(f'  saved to {path}')

print('Done.')
