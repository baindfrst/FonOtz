import requests
import codecs
from bs4 import BeautifulSoup as BS
import json

def getrefs(url):

    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
        'Connection': 'keep-alive',
        'Host': 'market.yandex.ru',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36',
    }
    #
    # response = requests.request("GET", url, headers=headers)
    #
    # readedContent = response.content

    with codecs.open("PhonesPage.html", "r", "utf_8_sig") as file:
        readedContent = file.read()

    soup = BS(readedContent, "lxml")
    # all_ph = soup.find_all("div", class_ = "_2im8- _2S9MU _2jRxX")
    ref = soup.find_all("div", class_="_3KhA2 _1jTgr")
    print(len(ref))
    ref_array = []

    for r in ref:
        item = r.find("a")
        url = item.get("href")
        if (url != None and (url not in ref_array)):
            ref_array.append(url)
    print("start saving ref")
    with open("fileRef.json", "w") as file:
        json.dump(ref_array, file)
    print("saving ref success")