import yake
import pandas as pd
from ast import literal_eval
def list_to_dict(listinp):
    out = dict()
    for item in listinp:
        out[item[0]] = item[1]
    return out
def yake_call():
    #чтение исходных отзывов
    dataframe = pd.read_csv("dataframe.csv")
    dataframe['positive'] = dataframe['positive'].apply(literal_eval)
    dataframe['negative'] = dataframe['negative'].apply(literal_eval)
    #создание таблицы для результатов
    df_yake = pd.DataFrame(columns=["model", "positive", "negative"])
    print("YAKE start")

    for model in dataframe["model"]:
        positive = ""

        #проход по отзывам для объединения положительных и отрицательных отзывов в один текст
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

        #создание экземпляра для извлечения ключевых слов и извлечение ключевых слов из положительных слов
        extractor = yake.KeywordExtractor(

            lan="ru",  # язык
            n=2,  # максимальное количество слов в фразе
            dedupLim=0.3,  # порог похожести слов
            top=10  # количество ключевых слов
        )
        pos = extractor.extract_keywords(positive)

        # создание экземпляра для извлечения ключевых слов и извлечение ключевых слов из отрицательных слов
        extractor = yake.KeywordExtractor(

            lan="ru",  # язык
            n=2,  # максимальное количество слов в фразе
            dedupLim=0.3,  # порог похожести слов
            top=10  # количество ключевых слов
        )
        neg = extractor.extract_keywords(negative)

        #занесение результатов в таблицу
        pos = list_to_dict(pos)
        neg = list_to_dict(neg)
        df_yake = df_yake._append({"model": model, "positive": list(pos.keys()), "negative": list(neg.keys())}, ignore_index=True)
    #сохранение таблицы в файл
    df_yake.to_csv("yake.csv")
