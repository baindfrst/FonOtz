import json
import requests
from bs4 import BeautifulSoup as BS
import pandas as pd
def request_to_page(url):
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

    response = requests.request("GET", url, headers=headers)
    readedContent = response.content
    soup = BS(readedContent, "lxml")
    ref = soup.find("div", class_="cENS_")
    if(ref == None):
        return "Allarm"
    return "https://market.yandex.ru" + ref.find("a").get("href")

def request_to_reviews(url):
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

    response = requests.request("GET", url, headers=headers)
    readedContent = response.content
    soup = BS(readedContent, "lxml")
    reviews_dict = {"positive": [], "negative": [], "comment": []}
    reviews_list_tag = soup.find("div", class_="_199Tg")
    if(reviews_list_tag != None):
        list_of_reviews_tags = reviews_list_tag.find_all("a")
        refs_to_comments = list()
        for ref_tag in list_of_reviews_tags:
            url = ref_tag.get("href")
            url = "https://market.yandex.ru/" + url
            if (url != None and url not in refs_to_comments):
                refs_to_comments.append(url)
        for url in refs_to_comments:
            response = requests.request("GET", url, headers=headers)
            readedContent = response.content
            soup = BS(readedContent, "lxml")
            reviews = soup.find_all("div", class_="_3IXcz")

            for review in reviews:
                types_of_rewiev = review.find_all("dl")
                for data in types_of_rewiev:
                    if data["data-auto"] == "review-pro":
                        reviews_dict["positive"].append(data.find("dd").text)
                    if data["data-auto"] == "review-contra":
                        reviews_dict["negative"].append(data.find("dd").text)
                    if data["data-auto"] == "review-comment":
                        reviews_dict["comment"].append(data.find("dd").text)
    else:
        reviews = soup.find_all("div", class_="_3IXcz")

        for review in reviews:
            types_of_rewiev = review.find_all("dl")
            for data in types_of_rewiev:
                if data["data-auto"] == "review-pro":
                    reviews_dict["positive"].append(data.find("dd").text)
                if data["data-auto"] == "review-contra":
                    reviews_dict["negative"].append(data.find("dd").text)
                if data["data-auto"] == "review-comment":
                    reviews_dict["comment"].append(data.find("dd").text)
        # reviews_dict["positive"] = types_of_rewiev.find("data-auto" = "review-pro").find("dd").text
        # reviews_dict["negative"] = types_of_rewiev.find({"data-auto": "review-contra"}).find("dd").text
        # reviews_dict["coment"] = types_of_rewiev.find({"data-auto": "review-comment"}).find("dd").text
    return reviews_dict

def getreviews(ref_array):
    print("searching reviews")
    dataframe = pd.DataFrame(columns=['model', 'positive', 'negative', 'comment'])
    # with open("reviews_file.txt", "w") as f:
    len_of_refs = len(ref_array[1:len(ref_array)])
    for i, ref in enumerate(ref_array[1:len(ref_array)]):
        flag_drope = False
        ref_to_reviews = request_to_page(ref)
        if ref_to_reviews == "Allarm":
            continue
        name_of_phone = ref[ref.find('ru') + 3:ref.find('/', ref.find('ru') + 4)]

        print(str(i) + "/" + str(len_of_refs) + " review on: " + name_of_phone)
        for i in dataframe["model"]:
            if name_of_phone in i:
                print("droped")
                flag_drope = True
                break
        if(flag_drope):
            continue
        review_dict_for_phone = request_to_reviews(ref_to_reviews)
        dataframe.loc[len(dataframe.index)] = [name_of_phone.strip(), review_dict_for_phone["positive"],
                                               review_dict_for_phone["negative"], review_dict_for_phone["comment"]]
        print(f"review to {name_of_phone} saved!!!")
        # f.write(name_of_phone)
        # for dict_key in review_dict_for_phone.keys():
        #     f.write("\n    " + dict_key + ":")
        #     for rev in review_dict_for_phone[dict_key]:
        #         f.write("\n            " + rev)
    dataframe.to_csv('dataframe.csv')
    print("MODELS saved")