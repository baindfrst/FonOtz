from yake import KeywordExtractor
from ast import literal_eval
import nltk
import json
from rake_nltk import Rake
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from jellyfish import jaro_similarity
nltk.download('stopwords')
import pymorphy3
def normalize_words(text):
    morph = pymorphy3.MorphAnalyzer()
    normalized_words = []
    for word in text.split():
        word = word.lower()
        word = word.strip('.,!?;:()[]{}"\'')
        p = morph.parse(word)[0]
        normalized_words.append(p.normal_form)
    return normalized_words

def extr_n_keywords(NUM_OF_KEYWORDS, path, name_folders):
    for file_name in name_foledrs.keys():
        for num_file in range(name_foledrs[file_name]):
            data_from_file = []
            with open(path + file_name + str(num_file) + ".jsonlines\\" + file_name + str(num_file) + ".jsonlines", "r",
                      encoding="utf-8") as file_text:
                data_from_file = [json.loads(obj) for obj in file_text]
            tf_idf_texts = []
            # with tqdm(total=len(data_from_file)) as pbar:
            #     for i, obj in enumerate(data_from_file):
            #         tf_idf_texts.append(" ".join(normalize_words(obj["content"])))
            #         pbar.update(1)
            #     with open(file_name + str(num_file) + "all_texts.json", "w") as file:
            #         json.dump(tf_idf_texts, file)
            # with open(file_name + str(num_file) + "all_texts.json", "r") as file:
            #     tf_idf_texts = json.load(file)

            data_result_bow = []
            data_result_tfidf = []
            data_result_rake = []
            data_result_yake = []
            with tqdm(total=len(data_from_file)) as pbar:
                for i, obj in enumerate(data_from_file):
                    # BOW
                    dict_werb = {}
                    tokens = normalize_words(obj["content"])
                    for token in tokens:
                        if token in dict_werb:
                            dict_werb[token] += 1
                        else:
                            dict_werb[token] = 1
                    data_result_bow.append(
                        [key for key, value in sorted(dict_werb.items(), key=lambda item: item[1])][:NUM_OF_KEYWORD])

                    # TF-IDF
                    text = " ".join(normalize_words(obj["content"]))
                    tfidf_vectorizer = TfidfVectorizer()
                    tfidf_matrix = tfidf_vectorizer.fit_transform(tf_idf_texts)
                    tfidf_scores = tfidf_matrix.toarray()[i]
                    feature_names = tfidf_vectorizer.get_feature_names_out()
                    data_result_tfidf.append(
                        [word for _, word in sorted(zip(tfidf_scores, feature_names), reverse=True)][:NUM_OF_KEYWORD])

                    # RAKE
                    stopwords_test = [
                        'и', 'в', 'на', 'о', 'с', 'из', 'для', 'как', 'который', 'это', 'быть', 'а', 'или', 'то', 'от',
                        'ныне', 'быть', 'не',
                        'так', 'он', 'я', 'мы', 'ты', 'оно', 'они', 'его', 'ее', 'её', 'их', 'может', 'быть', 'ли',
                        'бы', 'вы', 'ваш', 'вас',
                        'свои', 'такой', 'какой', 'где', 'когда', 'за', 'чей', 'чего', 'по', 'что', 'к', 'обо', 'у',
                        'над', 'под', 'перед',
                        'между', 'через', 'во', 'об', 'со', 'кто', 'чтобы', 'где', 'куда', 'откуда', 'когда', 'почему',
                        'как', 'чем', 'сколько',
                        'которые', 'этот', 'тот', 'такой', 'та', 'эта', 'эти', 'те', 'того', 'оном', 'оных', 'такой',
                        'такие', 'таким', 'такими',
                        'такого', 'таких', 'чей', 'чья', 'чье', 'чьи', 'чьем', 'чьими', 'чьего', 'чьих', 'мои', 'моего',
                        'моему', 'моим', 'моими',
                        'мое', 'моих', 'твой', 'твоего', 'твоему', 'твоим', 'твоими', 'твое', 'твоих', 'наш', 'нашего',
                        'нашему', 'нашим', 'нашими',
                        'наше', 'наших', 'ваш', 'вашего', 'вашему', 'вашим', 'вашими', 'ваше', 'ваших', 'он', 'она',
                        'оно', 'они', 'я', 'мы', 'ты',
                        'вы', 'мне', 'тебе', 'ему', 'ей', 'они', 'нам', 'вам', 'им', 'нас', 'вас', 'их', 'меня', 'тебя',
                        'себя', 'него', 'нее', 'них',
                        'мной', 'тобой', 'собой', 'им', 'ей', 'ними', 'мною', 'тобою', 'собою', 'им', 'ею', 'ними'
                    ]
                    punct_list = ['.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}', '\"', '\'', '«', '»', '–', '—', '…']
                    rake_base = Rake(stopwords=stopwords_test, punctuations=punct_list, language='ru', min_length=1, max_length=2, include_repeated_phrases = True)
                    rake_base.extract_keywords_from_text(obj["content"])
                    data_result_rake.append(rake_base.get_ranked_phrases()[:NUM_OF_KEYWORD])

                    # YAKE
                    extractor = KeywordExtractor(

                        lan="ru",  # язык
                        n=3,  # максимальное количество слов в фразе
                        dedupLim=0.9,  # порог похожести слов
                        top=NUM_OF_KEYWORD  # количество ключевых слов
                    )
                    data_result_yake_score = extractor.extract_keywords(obj["content"])
                    data_result_yake_keywords = []
                    for i in data_result_yake_score:
                        data_result_yake_keywords.append(i[0])
                    data_result_yake.append(data_result_yake_keywords)
                    pbar.update(1)
            print("_________________________________")
            print(data_result_bow)
            print(data_result_tfidf)
            print(data_result_rake)
            print(data_result_yake)

            with open(file_name + str(num_file) + "bow.json", "w") as file:
                json.dump(data_result_bow, file)
            with open(file_name + str(num_file) + "tfidf.json", "w") as file:
                json.dump(data_result_tfidf, file)
            with open(file_name + str(num_file) + "rake.json", "w") as file:
                json.dump(data_result_rake, file)
            with open(file_name + str(num_file) + "yake.json", "w") as file:
                json.dump(data_result_yake, file)

def f1_at_10_romal(ground_truths, predictions):
    num_queries = len(ground_truths)
    num_correct = 0
    num_predicted = 0
    num_relevant = 0
    for i in range(num_queries):
        ground_truth = []
        for word in ground_truths[i]:
            ground_truth.append(word.lower())
        prediction = []
        morph = pymorphy3.MorphAnalyzer()
        for word in predictions[i][:10]:
            prediction.append(word.lower())
            for gold_word in ground_truth:
                if morph.parse(word.lower())[0].normal_form == gold_word:
                    num_correct += 1
                    break
        num_predicted += len(prediction)
        num_relevant += min(len(ground_truth), 10)
    precision = num_correct / num_predicted if num_predicted > 0 else 0
    recall = num_correct / num_relevant if num_relevant > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return (f1, num_correct)

def f1_at_10(ground_truths, predictions, jaro_coef = 0.9):
    num_queries = len(ground_truths)
    num_correct = 0
    num_predicted = 0
    num_relevant = 0
    for i in range(num_queries):
        ground_truth = []
        for word in ground_truths[i]:
            ground_truth.append(word.lower())
        prediction = []
        for word in predictions[i][:10]:
            prediction.append(word.lower())
            for gold_word in ground_truth:
                if jaro_similarity(word.lower(), gold_word) >= jaro_coef:
                    num_correct += 1
                    break
        num_predicted += len(prediction)
        num_relevant += min(len(ground_truth), 10)
    precision = num_correct / num_predicted if num_predicted > 0 else 0
    recall = num_correct / num_relevant if num_relevant > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return (f1, num_correct)

#определение файлов из датасета
name_foledrs = {"cyberleninka_": 5, "habrahabr_": 4, "ng_": 2, "russia_today_": 8}
path = "ru_kw_eval_datasets-master\\data\\"

#количество ключевых слов для извлечения
NUM_OF_KEYWORD = 10

#извлечение ключевых слов. Извлеченные слова сохраняются в файл с названием "название датасета + название алгоритма для извлечения"
#не стоит повторно все это запускать, так как это занимает оооочень много времени, часа 2 точно
# extr_n_keywords(NUM_OF_KEYWORD, path, name_foledrs)
results = []

#подсчет мер для извлеченных слов
for file_name in name_foledrs.keys():
    for num_file in range(name_foledrs[file_name]):
        data_result_bow = []
        data_result_tfidf = []
        data_result_rake = []
        data_result_yake = []

        #загрузка извлеченных ранее слов из файлов
        with open(file_name + str(num_file) + "bow.json", "r") as file:
            data_result_bow = json.load(file)
        with open(file_name + str(num_file) + "tfidf.json", "r") as file:
            data_result_tfidf = json.load(file)
        with open(file_name + str(num_file) + "rake.json", "r") as file:
            data_result_rake = json.load(file)
        with open(file_name + str(num_file) + "yake.json", "r") as file:
            data_result_yake = json.load(file)
        data_from_file = []

        #загрузка предопределенных заранее ключевых слов из датасета
        with open(path + file_name + str(num_file) + ".jsonlines\\" + file_name + str(num_file) + ".jsonlines", "r",
                  encoding="utf-8") as file_text:
            data_from_file = [json.loads(obj) for obj in file_text]
        gold_keys = []
        total_count_keywords = 0
        for i, obj in enumerate(data_from_file):
            gold_keys.append(obj["keywords"])
            total_count_keywords += len(obj["keywords"])

        #подсчет F1@10 метрики для каждого алгоритма
        result_test = []
        result_test.append(f1_at_10(gold_keys, data_result_bow))
        result_test.append(f1_at_10(gold_keys, data_result_tfidf))
        result_test.append(f1_at_10(gold_keys, data_result_rake))
        result_test.append(f1_at_10(gold_keys, data_result_yake))
        results.append(result_test)
        print(f"{file_name + str(num_file)} f1@10 (total key_words - {total_count_keywords}): "
              f"\n\t bow {result_test[0][0]}, correct {result_test[0][1]}"
              f"\n\t tf_idf {result_test[1][0]}, correct {result_test[1][1]}"
              f"\n\t rake {result_test[2][0]}, correct {result_test[2][1]}"
              f"\n\t yake {result_test[3][0]}, correct {result_test[3][1]}")

#подсчет средней F1@10 метрики по всем файлам датасета
print(f"bow avg f1: {sum([x[0][0] for x in results])/len(results)}")
print(f"tf_idf avg f1: {sum([x[1][0] for x in results])/len(results)}")
print(f"rake avg f1: {sum([x[2][0] for x in results])/len(results)}")
print(f"yake avg f1: {sum([x[3][0] for x in results])/len(results)}")

#сохранение результата, полученного после подсчетов
with open("results.json", "w") as file:
    json.dump(results, file)



