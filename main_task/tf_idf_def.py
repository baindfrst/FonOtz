import json
import pandas as pd
import pymorphy3
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from ast import literal_eval


def normalize_words(text):
    morph = pymorphy3.MorphAnalyzer()
    normalized_words = []
    for word in text.split():
        word = word.lower()
        word = word.strip('.,!?;:()[]{}"\'')
        p = morph.parse(word)[0]
        normalized_words.append(p.normal_form)
    return " ".join(normalized_words)


def tf_idf():
    print("TF-IDF work start")

    #чтение исходных данных из файла
    dataframe = pd.read_csv("dataframe.csv")
    df_tf_idf = pd.DataFrame(columns=["model", "positive", "negative"])
    all_rews = list()
    dataframe['positive'] = dataframe['positive'].apply(literal_eval)
    dataframe['negative'] = dataframe['negative'].apply(literal_eval)
    dataframe['comment'] = dataframe['comment'].apply(literal_eval)

    #занесение всех отзывов в один текст для подсчета tf idf
    # for model in dataframe["model"]:
    #     str_pos = str()
    #     str_neg = str()
    #     str_com = str()
    #     for rev in dataframe.at[dataframe[dataframe["model"] == model].index[0], "positive"]:
    #         str_pos += " " + normalize_words(rev)
    #     for rev in dataframe.at[dataframe[dataframe["model"] == model].index[0], "negative"]:
    #         str_neg += " " + normalize_words(rev)
    #     for rev in dataframe.at[dataframe[dataframe["model"] == model].index[0], "comment"]:
    #         str_com += " " + normalize_words(rev)
    #     print(f"model - {model} comleted")
    #     all_rews.append(str_pos)
    #     all_rews.append(str_neg)
    #     all_rews.append(str_com)
    #     with open("all_rews_list.json", "w") as file:
    #         json.dump(all_rews, file)

    #чтение всех отзывов в виде одного текста
    with open("all_rews_list.json", "r") as file:
        all_rews = json.load(file)

    #подсчет весов tf_idf для всех слов
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_rews)

    #извлечение ключевых слов
    feature_names = tfidf_vectorizer.get_feature_names_out()
    for i in range(len(tfidf_matrix.toarray()) // 2):
        tfidf_scores_pos = tfidf_matrix.toarray()[i]
        tfidf_scores_neg = tfidf_matrix.toarray()[i + len(tfidf_matrix.toarray()) // 2]

        #сортировка ключевых слов
        sorted_keywords_pos = str([(word, _) for _, word in sorted(zip(tfidf_scores_pos, feature_names), reverse=True)])
        sorted_keywords_neg = str([(word, _) for _, word in sorted(zip(tfidf_scores_neg, feature_names), reverse=True)])
        model = dataframe["model"].iloc[[i]]
        #запись данных в таблицу
        df_tf_idf = df_tf_idf._append({"model": model.to_string(index = False), "positive": sorted_keywords_pos, "negative": sorted_keywords_neg}, ignore_index=True)
    #сохранение данных в файл
    df_tf_idf.to_csv("tf_idf.csv")