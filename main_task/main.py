import requests
import codecs
from bs4 import BeautifulSoup as BS
import json
from yake_my import yake_call
from RAKE import RAKE_call
import pandas as pd
from tf_idf_def import tf_idf
from pymystem3 import Mystem
from string import punctuation
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from bag_of_werb import bag_of_werbs
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from getrefs import getrefs
from getreviews import getreviews
import pymorphy3
from rake_nltk import Rake

def preprocess_text(text):
    morph = pymorphy3.MorphAnalyzer()
    normalized_words = []
    for word in text.split():
        word = word.lower()
        word = word.strip('.,!?;:()[]{}"\'')
        p = morph.parse(word)[0]
        normalized_words.append(p.normal_form)
    return " ".join(normalized_words)

pd.options.display.max_rows = 20
pd.options.display.max_columns = 5

ref_array = list()
#взяите ссылок с сайта
getrefs("https://market.yandex.ru/catalog--smartfony/26893750/list?hid=91491&local-offers-first=0&glfilter=21194330%3A38083630%2C34066443")
# загрузка из файл ссылок, что были получены с сайта
with open("fileRef.json", "r") as file:
     ref_array = json.load(file)

#взятие отзыва со списка ссылок на продукты
#getreviews(ref_array) #для запуска это части кода нужно убрать коментарий. Данные записывает в файл dataframe.csv
dataframe = pd.read_csv("dataframe.csv")


len_of_refs = len(ref_array[1:len(ref_array)])

#
# выделение из отзывов сущ + прил
#

# noun = "NOUN"
# adjf = "ADJF"
#
# dict_reviews_noun_adjf = dict()
#
# for i, dict_key_frst in enumerate(dict_all_reviews):
#     dict_reviews_noun_adjf[dict_key_frst] = dict()
#     for dict_key in dict_all_reviews[dict_key_frst]:
#         dict_reviews_noun_adjf[dict_key_frst][dict_key] = list()
#         print(str(i) + "/" + str(len_of_refs) + " proc reviews: " + dict_key_frst + " " + dict_key)
#         for rev in dict_all_reviews[dict_key_frst][dict_key]:
#             rev_splited = rev.split()
#             for i in range(1, len(rev_splited)):
#                 if(morph.parse(rev_splited[i])[0].tag.POS == noun and morph.parse(rev_splited[i - 1])[0].tag.POS == adjf or
#                 morph.parse(rev_splited[i])[0].tag.POS == adjf and morph.parse(rev_splited[i])[0].tag.POS == noun):
#                     if(morph.parse(rev_splited[i - 1])[0].tag.POS == adjf):
#                         dict_reviews_noun_adjf[dict_key_frst][dict_key].append({"adjf": rev_splited[i - 1] , "noun": rev_splited[i]})
#                     else:
#                         dict_reviews_noun_adjf[dict_key_frst][dict_key].append({"adjf": rev_splited[i] , "noun": rev_splited[i - 1]})
# print("start saving reviews adjf + noun")
# with open("reviews_file_pymorphy.json", "w") as f: #{"model": {"positiv": ({"adjf": "", "noun": ""}, ...), "negativ": (...), "comment": (...)}, ...}
#     json.dump(dict_reviews_noun_adjf, f)
# print("saving reviews adjf + noun success")



#
# Bag-of_Werbs
#

#bag_of_werbs() #внутри функции читаются данные из dataframe.csv. реализует алгоритм мешка слов. выходые данные выдает в файл bag_of_werb.csv

#
#tf-idf
#

#tf_idf() #внутри функции читаются данные из dataframe.csv. реализует алгоритм с tf_idf. выходые данные выдает в файл tf_idf.csv

#
#RAKE
#

#RAKE_call() #внутри функции читаются данные из dataframe.csv. реализует алгоритм RAKE. выходые данные выдает в файл RAKE.csv

#
# YAKE!
#
#yake_call() #внутри функции читаются данные из dataframe.csv. реализует алгоритм YAKE. выходые данные выдает в файл yake.csv

#
#формирование единой таблицы со всеми результатами
#

#создание таблицы
pd.set_option('display.max_colwidth', None)
df_all = pd.DataFrame(columns=["model", "positive rew", "positive BoW", "positive TF_IDF",
                               "positive RAKE", "positive YAKE!", "negative rew",
                               "negative BoW", "negative TF_IDF",
                               "negative RAKE", "negative YAKE!"])

#чтение данных от каждого алгоритма
df_market = pd.read_csv("dataframe.csv")
df_BoW = pd.read_csv("bag_of_werb.csv")
df_tf_idf = pd.read_csv("tf_idf.csv")
df_rake = pd.read_csv("rake.csv")
df_yake = pd.read_csv("yake.csv")

#превидение данных к нужному формату для итерации
df_market['positive'] = df_market['positive'].apply(literal_eval)
df_market['negative'] = df_market['negative'].apply(literal_eval)
df_BoW['positive'] = df_BoW['positive'].apply(literal_eval)
df_BoW['negative'] = df_BoW['negative'].apply(literal_eval)
df_tf_idf['positive'] = df_tf_idf['positive'].apply(literal_eval)
df_tf_idf['negative'] = df_tf_idf['negative'].apply(literal_eval)
df_rake['positive'] = df_rake['positive'].apply(literal_eval)
df_rake['negative'] = df_rake['negative'].apply(literal_eval)
df_yake['positive'] = df_yake['positive'].apply(literal_eval)
df_yake['negative'] = df_yake['negative'].apply(literal_eval)

#заполнение финальной таблицы
for model in df_market["model"]:
    df_all = df_all._append({"model": model, "positive rew": df_market[df_market["model"] == model]["positive"].to_string(),
                             "positive BoW": df_BoW[df_BoW["model"] == model]["positive"].to_string(),
                             "positive TF_IDF": df_tf_idf[df_tf_idf["model"] == model]["positive"].to_string(),
                             "positive RAKE": df_rake[df_rake["model"] == model]["positive"].to_string(),
                             "positive YAKE!": df_yake[df_yake["model"] == model]["positive"].to_string(),
                             "negative rew": df_market[df_market["model"] == model]["negative"].to_string(),
                             "negative BoW": df_BoW[df_BoW["model"] == model]["negative"].to_string(),
                             "negative TF_IDF": df_tf_idf[df_tf_idf["model"] == model]["negative"].to_string(),
                             "negative RAKE": df_rake[df_rake["model"] == model]["negative"].to_string(),
                             "negative YAKE!": df_yake[df_yake["model"] == model]["negative"].to_string()}, ignore_index=True)
    print(df_tf_idf[df_tf_idf["model"] == model])

#сохранение результатов в csv и excel
df_all.to_csv("rezult.csv")
df_all.to_excel("rezult.xlsx", index=False)