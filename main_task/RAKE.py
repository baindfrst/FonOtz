import nltk
nltk.download('stopwords')
import pymorphy3
from rake_nltk import Rake
import pandas as pd
from nltk.corpus import stopwords
from ast import literal_eval

def RAKE_call():
    print("RAKE start")

    #загрузка изначальных отзывов
    dataframe = pd.read_csv("dataframe.csv")
    dataframe['positive'] = dataframe['positive'].apply(literal_eval)
    dataframe['negative'] = dataframe['negative'].apply(literal_eval)
    morph = pymorphy3.MorphAnalyzer()

    #создание таблицы для результатов
    df_rake = pd.DataFrame(columns=["model", "positive", "negative"])

    #загрузка списка стоп слов из nltk
    stop_words = set(stopwords.words('russian'))
    for model in dataframe["model"]:
        print(model)
        positive = ""

        #вставка точек в конце отзывов, тк алгоритм чувствителен к ним, а так же объединение всех отзывов в один текст
        for rew in dataframe.at[dataframe[dataframe["model"] == model].index[0], "positive"]:
            if(len(positive) > 0):
                if (positive.strip()[-1] != "."):
                    positive += '.'
            positive = " " + rew.strip()
        negative = ""

        for rew in dataframe.at[dataframe[dataframe["model"] == model].index[0], "negative"]:
            if(len(negative) > 0):
                if (positive.strip()[-1] != "."):
                    negative += '.'
            negative += " " + rew.strip()

        #создание экземпляров класса алгоритма и извлечение ключевых слов

        #извлечение ключевых слов из положительных отзывов
        rake_base = Rake(stopwords=stop_words, language='ru', min_length=1, max_length=2)
        rake_base.extract_keywords_from_text(positive)
        pos = rake_base.get_ranked_phrases()
        #извлечение ключевых слов из отрицательных слов
        rake_base = Rake(stopwords=stop_words, language='ru', min_length=1, max_length=2)
        rake_base.extract_keywords_from_text(negative)
        neg = rake_base.get_ranked_phrases()

        #занесение результатов в таблицу
        df_rake = df_rake._append({"model": model, "positive": pos, "negative": neg}, ignore_index=True)
    #сохранение результатов в файл
    df_rake.to_csv("rake.csv")
