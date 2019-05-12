# Modified from https://github.com/BenjiKCF/Neural-Net-with-Financial-Time-Series-Data/blob/master/1.%20NY_Times_downloader.ipynb

import requests
import json
from pathlib import Path
import time

api = "uwli6OeGx45lNnNGGCtEKtv9hlT9lnzi"

def download_json(year, month, api):
    "Download news for a particular year and month and save as a json file"
    
    url = "http://api.nytimes.com/svc/archive/v1/{}/{}.json?api-key={}"
    url = url.format(year, month, api)
    
    file_str = 'jsons/' + str(year) + '-' + '{:02}'.format(month) + '.json'
    print(file_str)

    if Path(file_str).is_file():
        print('Skipped')
        return

    items = requests.get(url)
    
    data = items.json()

    with open(file_str, 'w') as f:
        json.dump(data, f)

    time.sleep(6)
    
    return "Finished downloading {}/{}".format(year, month)

for year in range(2000, 2019, 1):
    for month in range(1, 13, 1):
        download_json(year, month, api)

download_json(2019, 1, api)
download_json(2019, 2, api)
download_json(2019, 3, api)
download_json(2019, 4, api)
download_json(2019, 5, api)

