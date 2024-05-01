import json
import chardet
from yake import KeywordExtractor
from jellyfish import jaro_similarity
from operator import itemgetter
from sklearn.metrics import f1_score
import os
import time
from nltk.corpus import stopwords
from rake_nltk import Rake
import threading
import nltk
nltk.download('stopwords')
nltk.download('punkt')

#подсчет F1@10 метрики с мерой Джаро
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
    return f1

def check_word_in_list_juc(word, list_to_check, coef):
    for word_gold in list_to_check:
        if jaro_similarity(word, word_gold) >= coef:
            return True
    return False
def average_precision(ground_truth, predictions, coef_juc):
    num_hits = 0
    num_predictions = 0
    avg_prec = 0.0
    for i, pred in enumerate(predictions):
        if check_word_in_list_juc(pred, ground_truth, coef_juc):
            num_hits += 1
            num_predictions += 1
            if num_predictions == 1:
                precision = 1.0
            else:
                precision = num_hits / num_predictions
            recall = num_hits / len(ground_truth)
            if recall < 1.0:
                avg_prec += precision
    if len(ground_truth) == 0:
        return 0.0
    else:
        return avg_prec / len(ground_truth)

#функция для подсчета MAP@50 метрики
def map_at_50(ground_truths, predictions, coef):
    num_queries = len(ground_truths)
    avg_prec = 0.0
    for i in range(num_queries):
        avg_prec += average_precision(ground_truths[i], predictions[i][:50], coef)
    return avg_prec / num_queries

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        return result['encoding']
def get_keywords_from_folder(dir, file_name_keyword, file_name_gold_keys, top_w, stop_words):
    rake_base = Rake(stopwords=stop_words, min_length=1, max_length=3)
    extractor = KeywordExtractor(lan="en", n=3, dedupLim=0.9, dedupFunc="seqm", top=top_w, features=None, windowsSize=1)
    name_texts_keys_file = []
    directory = dir
    count = 0

    for entry in os.scandir(directory):
        if entry.is_file() and entry.name.endswith('.txt'):
            name_texts_keys_file.append(entry.name.split('.')[0])
            count += 1
    directory = dir + "\\"
    gold_standard = []
    # список извлеченных ключевых слов для каждого текста
    keywords = []
    keywords_Rake = []
    print(dir)

    for i, file_name in enumerate(name_texts_keys_file):
        print(f"{dir} {top_w}: {i + 1}/{count}")
        file = open(directory + file_name + ".txt", "r", encoding=detect_encoding(directory + name_texts_keys_file[i] + ".txt"))
        text_arr = file.readlines()
        text = ""
        for elem in text_arr:
            text += elem
        keywords.append(extractor.extract_keywords(text))
        rake_base.extract_keywords_from_text(text)
        keywords_Rake.append(rake_base.get_ranked_phrases()[:top_w])
        file = open(directory + file_name + ".key", "r")
        keys = file.readlines()
        for i in range(len(keys)):
            if (dir != "Keyword-Extraction-Datasets-master\wiki20\documents"):
                keys[i] = keys[i].replace("\n", '')
            else:
                keys[i] = keys[i].replace("\n", '')
                keys[i] = keys[i].split(":")[1].strip()
        gold_standard.append(keys)
    with open(file_name_keyword, "w+") as file:
        json.dump(keywords, file)
    with open(file_name_keyword.replace("YAKE", "Rake"), "w+") as file:
        json.dump(keywords_Rake, file)
    with open(file_name_gold_keys, "w+") as file:
        json.dump(gold_standard, file)
#text search

#список, в котором описан относительный путь до директории с текстами и ключевыми словами для них, а так же название фалйов для извлеченных ключевы слов и ключевых слов, что должны быть для n-ого текста
file_names_list = [["Keyword-Extraction-Datasets-master\Krapivin2009\Krapivin2009\Data", "YAKE_Krapilov_keywords_must.json", "YAKE_Krapilov_keywords_extr.json"],
                   ["Keyword-Extraction-Datasets-master\WWW\WWW", "YAKE_WWW_keywords_must.json","YAKE_WWW_keywords_extr.json"],
                   ["Keyword-Extraction-Datasets-master\KDD\KDD", "YAKE_KDD_keywords_must.json","YAKE_KDD_keywords_extr.json"],
                   ["Keyword-Extraction-Datasets-master\SemEval2010\SemEval2010\SemEval-all", "YAKE_SemEval2010_keywords_must.json","YAKE_SemEval2010_keywords_extr.json"],
                   ["Keyword-Extraction-Datasets-master\wiki20\documents", "YAKE_wiki20_keywords_must.json","YAKE_wiki20_keywords_extr.json"],
                   ["Keyword-Extraction-Datasets-master\\theses100\\text", "YAKE_theses100_keywords_must.json","YAKE_theses100_keywords_extr.json"],
                   ["Keyword-Extraction-Datasets-master\\fao30\\documents", "YAKE_fao30_keywords_must.json","YAKE_fao30_keywords_extr.json"],
                   ["Keyword-Extraction-Datasets-master\\fao780", "YAKE_fao780_keywords_must.json","YAKE_fao780_keywords_extr.json"]] #citeulike180 просто ужасная организация файлов, потом сделаю, если понадобится
start = time.time()
threads = []

#извлечение ключевых слов (занимает кучу времени, около 3 часов, не стоит запускать эту часть просто так повторно)
# for list_call in file_names_list[:7]:
#     stop_words = set(stopwords.words('english'))
#     my_thread = threading.Thread(target=get_keywords_from_folder, args=(list_call[0], list_call[2], list_call[1], 10, stop_words))
#     my_thread.start()
#     threads.append(my_thread)
#     stop_words = set(stopwords.words('english'))
#     my_thread = threading.Thread(target=get_keywords_from_folder, args=(list_call[0], list_call[2].split(".")[0]+"_50top.json", list_call[1].split(".")[0]+"_50top.json", 50, stop_words))
#     my_thread.start()
#     threads.append(my_thread)
# for thread in threads:
#     thread.join()

JARO_COEF = 1 #кеф для меры Джаро

#подсчет средних мер для каждого датасета
for list_call in file_names_list[:7]:

    print("\n")
    #F1@10 для алгоритма YAKE
    gold_standard = []
    keywords = []
    # чтение извлеченных и определенных заранее слов
    with open(list_call[2], "r+") as file:
        keywords = json.load(file)
    with open(list_call[1], "r+") as file:
        gold_standard = json.load(file)
    just_keywords = []
    for set_k in keywords:
        new_pull = []
        for word_par in set_k:
            new_pull.append(word_par[0])
        just_keywords.append(new_pull)
    print(f"file dir: {list_call[0]}, avg YAKE F1 metric is: {f1_at_10(gold_standard, just_keywords, JARO_COEF)}")
    with open("out_put.txt", "a") as file:
        file.write(f"file dir: {list_call[0]}, avg YAKE F1 metric is: {f1_at_10(gold_standard, just_keywords, JARO_COEF)}\n")

    #F1@10 мера для алгоритма RAKE
    gold_standard = []
    keywords = []
    # чтение извлеченных и определенных заранее слов
    with open(list_call[2].replace("YAKE", "Rake"), "r+") as file:
        keywords = json.load(file)
    with open(list_call[1], "r+") as file:
        gold_standard = json.load(file)
    print(
        f"file dir: {list_call[0]}, avg RAKE F1 metric is: {f1_at_10(gold_standard, keywords, JARO_COEF)}")
    with open("out_put.txt", "a") as file:
        file.write(
            f"file dir: {list_call[0]}, avg RAKE F1 metric is: {f1_at_10(gold_standard, just_keywords, JARO_COEF)}\n")


    #MAP@50 мера для алгоритма YAKE
    gold_standard = []
    keywords = []
    # чтение извлеченных и определенных заранее слов
    with open(list_call[2].split(".")[0]+"_50top.json", "r+") as file:
        keywords = json.load(file)
    with open(list_call[1].split(".")[0]+"_50top.json", "r+") as file:
        gold_standard = json.load(file)
    just_keywords = []
    for set_k in keywords:
        new_pull = []
        for word_par in set_k:
            new_pull.append(word_par[0])
        just_keywords.append(new_pull)
    print(f"file dir: {list_call[0]}, Yake MAP@50 metric is: {map_at_50(gold_standard, just_keywords, JARO_COEF)}")
    with open("out_put.txt", "a") as file:
        file.write(f"file dir: {list_call[0]}, Yake MAP@50 metric is: {map_at_50(gold_standard, just_keywords, JARO_COEF)} \n")

    #MAP@50 мера для алгоритма RAKE

    #чтение извлеченных и определенных заранее слов
    with open(list_call[2].split(".")[0].replace("YAKE", "Rake")+"_50top.json", "r+") as file:
        keywords = json.load(file)
    with open(list_call[1].split(".")[0]+"_50top.json", "r+") as file:
        gold_standard = json.load(file)
    print(f"file dir: {list_call[0]}, Yake MAP@50 metric is: {map_at_50(gold_standard, keywords, JARO_COEF)}")
    with open("out_put.txt", "a") as file:
        file.write(f"file dir: {list_call[0]}, Yake MAP@50 metric is: {map_at_50(gold_standard, keywords, JARO_COEF)} \n")

print("время работы", time.time() - start)

