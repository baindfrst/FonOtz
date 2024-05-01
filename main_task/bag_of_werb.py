import nltk
import pandas as pd

nltk.download('stopwords')
from ast import literal_eval
from nltk.corpus import stopwords
import json
import pymorphy3
from pymystem3 import Mystem
from string import punctuation


def normalize_words(text):
    morph = pymorphy3.MorphAnalyzer()
    normalized_words = []
    for word in text.split():
        word = word.lower()
        word = word.strip('.,!?;:()[]{}"\'')
        p = morph.parse(word)[0]
        normalized_words.append(p.normal_form)
    return " ".join(normalized_words)
def bag_of_werbs():
    with open("fileRef.json", "r") as file:
        ref_array = json.load(file)

    print("BOW start")

    #чтение исходных отзывов
    dataframe = pd.read_csv("dataframe.csv")
    dataframe['positive'] = dataframe['positive'].apply(literal_eval)
    dataframe['negative'] = dataframe['negative'].apply(literal_eval)

    #создание таблицы для результатов
    df_bag = pd.DataFrame(columns=['model', 'positive', 'negative', 'count_rev'])
    stop_words_abow = dict()


    for model in dataframe["model"]:
        print(f"{model} in procces ///")
        #досоздание таблицы
        df_bag.loc[len(df_bag.index)] = [model, [],
                                               [], 0]
        df_bag.at[dataframe[dataframe["model"] == model].index[0], "count_rev"] = len(dataframe.at[dataframe[dataframe["model"] == model].index[0], "positive"]) + len(dataframe.at[dataframe[dataframe["model"] == model].index[0], "negative"])
        df_bag.at[dataframe[dataframe["model"] == model].index[0], "positive"] = dict()
        stop_words_abow[model] = list()

        #проход по всем отзывам положительным (rev)
        for rev in dataframe.at[dataframe[dataframe["model"] == model].index[0], "positive"]:
            #нормализация слов
            rev_splited = normalize_words(rev).split()

            for werb in rev_splited:
                if(len(werb) <= 3): #отбрасывание слов длинной меньше или равных 3 как стоп слов
                    continue

                #заполнения словаря {слово: его кол-во в отзывах
                if(werb in df_bag.at[dataframe[dataframe["model"] == model].index[0], "positive"].keys()):
                    df_bag.at[dataframe[dataframe["model"] == model].index[0], "positive"][werb] += 1
                else:
                    df_bag.at[dataframe[dataframe["model"] == model].index[0], "positive"][werb] = 1

        #создание словаря для отрицательных отзывов
        df_bag.at[dataframe[dataframe["model"] == model].index[0], "negative"] = dict()

        #проход по отрицательным отзывам
        for rev in dataframe.at[dataframe[dataframe["model"] == model].index[0], "negative"]:
            #нормализация слов
            rev_splited = normalize_words(rev).split()

            for werb in rev_splited:
                #отбрасывание слов, которые так же входят в положительные отзывы и удаление их из словаря положительных отзывов
                if(werb in df_bag.at[dataframe[dataframe["model"] == model].index[0], "positive"].keys() or len(werb) <= 3):
                    stop_words_abow[model].append(werb)
                    df_bag.at[dataframe[dataframe["model"] == model].index[0], "positive"].pop(werb, None)
                    continue

                #заполнение словаря отрицательных отзывов аналогично положительным
                if (werb in df_bag.at[dataframe[dataframe["model"] == model].index[0], "negative"].keys()):
                    df_bag.at[dataframe[dataframe["model"] == model].index[0], "negative"][werb] += 1
                else:
                    df_bag.at[dataframe[dataframe["model"] == model].index[0], "negative"][werb] = 1

        #сортировка словарей
        df_bag.at[dataframe[dataframe["model"] == model].index[0], "positive"] = sorted(df_bag.at[dataframe[dataframe["model"] == model].index[0], "positive"].items(), key=lambda item: item[1], reverse=True)
        df_bag.at[dataframe[dataframe["model"] == model].index[0], "negative"] = sorted(df_bag.at[dataframe[dataframe["model"] == model].index[0], "negative"].items(), key=lambda item: item[1], reverse=True)

    #сохранение результата
    print(df_bag)
    df_bag.to_csv("bag_of_werb.csv")